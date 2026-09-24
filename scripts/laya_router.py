"""Route second-turn KV reuse with local LAYA and measured cache costs."""

import argparse
import json
import secrets
import statistics
import subprocess
import time
from pathlib import Path

from benchmark import check_prompt, messages
from cache_policies import SALTS
from compressed_reuse import disk_usage, metrics, passed, save, server_info
from jev_router import load_cases, wait_for_idle, wait_for_seed
from two_turn import MODEL, chat

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "laya_router.json"
SUMMARY = ROOT / "results" / "laya_router_summary.json"
EVALUATION = ROOT / "results" / "laya_semantic_evaluation.json"
L2 = {mode: ROOT / "results" / "laya_l2" / mode for mode in SALTS}
MODEL_REVISION = "aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8"
MODEL_SNAPSHOT = (
    Path.home() / ".cache" / "huggingface" / "hub" /
    "models--convaiinnovations--laya" / "snapshots" / MODEL_REVISION
)
QUESTIONS = {
    "task": {
        "type": "choice",
        "instructions": "What work does the current question primarily require?",
        "criteria": {
            "lookup": "Retrieve one or a few explicit facts.",
            "summary": "Summarize supplied information.",
            "comparison": "Compare information from different parts of context.",
            "reasoning": "Use multi-step reasoning or deduction.",
            "exact": "Reproduce exact wording, quotations, numbers, or formatting.",
            "other": "None of these clearly apply.",
        },
    },
    "sensitivity": {
        "type": "score",
        "instructions": "How likely is a small error from compressing prior context to change whether the answer satisfies the current question?",
        "criteria": [
            "Low: a broad or simple factual answer tolerates small context errors.",
            "Medium: details matter, but ordinary compression may suffice.",
            "High: exact wording, exceptions, or several constraints must be preserved.",
        ],
    },
    "multi": {
        "type": "noul",
        "instructions": "Does this question require satisfying multiple independent instructions or constraints simultaneously?",
    },
}
SEMANTIC_PROBES = [
    ("Which city hosts the headquarters?", "low"),
    ("Who founded the company?", "low"),
    ("What year was the product launched?", "low"),
    ("Summarize the document in one sentence.", "medium"),
    ("Compare the founder's plans in two sections.", "medium"),
    ("How did the budget change between years?", "medium"),
    ("Describe the overall argument and its conclusion.", "medium"),
    ("What exact phrase did the founder use?", "high"),
    ("Quote the clause verbatim, including punctuation.", "high"),
    ("Compare four exceptions, preserve each qualifier, and quote them exactly.", "high"),
    ("List every deadline and its exception, then identify any conflict.", "high"),
    ("Reproduce the table with precisely the original row order.", "high"),
]


def load_laya():
    import laya

    if laya.__version__ != "0.3.11":
        raise RuntimeError(f"Expected laya 0.3.11, got {laya.__version__}")
    if not (MODEL_SNAPSHOT / "typed-decisions" / "model.safetensors").exists():
        raise RuntimeError(f"Download the pinned LAYA checkpoint first: {MODEL_SNAPSHOT}")
    return laya.load(str(MODEL_SNAPSHOT), subfolder="typed-decisions", device="cpu")


def judge(agent, question, prefix_tokens):
    state = {
        "question": question,
        "prefix_tokens": prefix_tokens,
        "num_system_instructions": 1,
        "response_format": "free_text",
    }
    started = time.perf_counter()
    result = agent.predict(state, QUESTIONS)
    return result, round(time.perf_counter() - started, 4)


def gpu_utilization():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        )
        return int(output.splitlines()[0].strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def probe_case(target, tokenizer):
    case = {
        "id": f"probe-{target}",
        "company": f"Calibration Works {target}",
        "founder": "Robin Vale",
        "question": "What color flag is mentioned in section 3?",
        "expected": ["orange"],
    }
    count = max(20, target // 42)
    for _ in range(8):
        sections = [f"Document for {case['company']}."]
        for number in range(1, count + 1):
            body = ("The flag was orange." if number == 3 else
                    "The team checked its schedule, inventory, and review notes.")
            sections.append(f"Section {number}: {body}")
        case["document"] = "\n".join(sections)
        case["seed_token_ids"] = tokenizer.apply_chat_template(
            messages(case, False), tokenize=True, add_generation_prompt=True
        )["input_ids"]
        case["query_token_ids"] = tokenizer.apply_chat_template(
            messages(case, True), tokenize=True, add_generation_prompt=True
        )["input_ids"]
        delta = target - len(case["query_token_ids"])
        if abs(delta) <= 80:
            break
        count = max(20, count + round(delta / 25))
    return case


def matched_tokens(case):
    seed, query = case["seed_token_ids"], case["query_token_ids"]
    common = 0
    for left, right in zip(seed, query):
        if left != right:
            break
        common += 1
    return common // 256 * 256


def check_server():
    pid, connected = server_info()
    if not connected:
        raise RuntimeError("Start vLLM with the JevLMCacheMPConnector")
    args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    config = json.loads(args[args.index(b"--kv-transfer-config") + 1])
    if config.get("kv_connector") != "JevLMCacheMPConnector":
        raise RuntimeError("The fresh route requires JevLMCacheMPConnector")
    return pid


def request(case, route, endpoint, metrics_url):
    before = wait_for_idle(metrics_url)
    before_disk = {mode: disk_usage(path)["bytes"] for mode, path in L2.items()}
    salt = SALTS.get(route, secrets.token_urlsafe(32))
    started = time.perf_counter()
    result = chat(endpoint, messages(case, True), cache_salt=salt)
    result["request_seconds"] = round(time.perf_counter() - started, 4)
    check_prompt(result, case, True)
    settled = wait_for_idle(metrics_url)
    result["l1_hit_tokens"] = round(settled["l1"] - before["l1"])
    result["l2_hit_tokens"] = round(settled["l2"] - before["l2"])
    result["l2_store_submitted_chunks"] = round(settled["submitted"] - before["submitted"])
    result["l2_bytes_added"] = {
        mode: disk_usage(path)["bytes"] - before_disk[mode] for mode, path in L2.items()
    }
    result["gold_pass"] = passed(result["answer"], case["expected"])
    if route in SALTS:
        if result["l2_hit_tokens"] <= 0 or result["l1_hit_tokens"] != 0:
            raise RuntimeError(f"{case['id']}: {route} did not retrieve only from L2: {result}")
        wrong = "tq4" if route == "fp8" else "fp8"
        if result["l2_bytes_added"][wrong]:
            raise RuntimeError(f"{case['id']}: {route} wrote to the {wrong} adapter")
    elif (result["l1_hit_tokens"] or result["l2_hit_tokens"] or
          result["l2_store_submitted_chunks"] or any(result["l2_bytes_added"].values())):
        raise RuntimeError(f"{case['id']}: fresh reused or stored KV: {result}")
    return result


def estimate(calibration, route, tokens):
    samples = sorted(calibration[route], key=lambda row: row["matched_tokens"])
    if tokens <= samples[0]["matched_tokens"]:
        return samples[0]["ttft_seconds"] * tokens / samples[0]["matched_tokens"]
    if tokens >= samples[-1]["matched_tokens"]:
        return samples[-1]["ttft_seconds"] * tokens / samples[-1]["matched_tokens"]
    for low, high in zip(samples, samples[1:]):
        if low["matched_tokens"] <= tokens <= high["matched_tokens"]:
            fraction = (tokens - low["matched_tokens"]) / (
                high["matched_tokens"] - low["matched_tokens"]
            )
            return low["ttft_seconds"] + fraction * (
                high["ttft_seconds"] - low["ttft_seconds"]
            )
    raise AssertionError("No calibration samples")


def decide(costs, laya_result, storage_pressure, args):
    if costs["matched_tokens"] == 0 or not costs["available"]["fp8"]:
        return "fresh", "no matching FP8 cache"
    if costs["fp8"] > args.fp8_margin * costs["fresh"]:
        return "fresh", "FP8 load estimate exceeds prefill margin"
    if laya_result is None:
        return "fresh", "LAYA unavailable"
    answers = laya_result["answers"]
    high = answers["sensitivity"]["probabilities"]["2"]
    low = answers["sensitivity"]["probabilities"]["0"]
    multi = answers["multi"]["noul"]
    lookup = answers["task"]["probabilities"]["lookup"]
    if high >= args.high_probability or multi >= args.multi_probability:
        return "fresh", "high semantic sensitivity"
    if (storage_pressure and costs["available"]["tq4"] and
            low >= args.low_probability and lookup >= args.lookup_probability and
            costs["tq4"] <= args.tq4_margin * costs["fresh"]):
        return "tq4", "low-sensitivity lookup under storage pressure"
    return "fp8", "eligible FP8 default"


def seed(cases, endpoint, metrics_url):
    from transformers import AutoTokenizer

    pid = check_server()
    if STATE.exists():
        raise RuntimeError(f"Remove generated state before reseeding: {STATE}")
    for path in L2.values():
        if path.exists() and any(path.iterdir()):
            raise RuntimeError(f"Use an empty L2 directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    probes = [probe_case(length, tokenizer) for length in (2048, 4096, 8192)]
    expected = 0
    before = metrics(metrics_url)
    for case in [*probes, *cases]:
        for mode, salt in SALTS.items():
            result = chat(endpoint, messages(case, False), cache_salt=salt)
            check_prompt(result, case, False)
            expected += len(case["seed_token_ids"]) // 256
            wait_for_seed(before, expected, case["id"], metrics_url)
            print(f"seed {case['id']} {mode}: {expected} chunks", flush=True)
    sizes = {mode: disk_usage(path) for mode, path in L2.items()}
    if any(not item["files"] for item in sizes.values()):
        raise RuntimeError(f"Missing codec L2 data: {sizes}")
    save({
        "model": MODEL, "laya_revision": MODEL_REVISION,
        "seed_server_pid": pid, "probes": probes,
        "seed_chunks": expected, "seed_l2": sizes,
    }, STATE)
    print(json.dumps({"seeded_cases": len(cases), "probes": len(probes),
                      "chunks": expected, "l2": sizes}, indent=2))


def calibrate(endpoint, metrics_url):
    state = json.loads(STATE.read_text())
    pid = check_server()
    if pid == state["seed_server_pid"]:
        raise RuntimeError("Restart vLLM after seeding; keep LMCache running")
    if "calibration" in state:
        raise RuntimeError("Calibration already completed")
    calibration = {route: [] for route in ("fresh", "fp8", "tq4")}
    for case in state["probes"]:
        for route in calibration:
            result = request(case, route, endpoint, metrics_url)
            calibration[route].append({
                "matched_tokens": matched_tokens(case),
                "ttft_seconds": result["ttft_seconds"],
                "l2_hit_tokens": result["l2_hit_tokens"],
                "answer": result["answer"],
            })
            print(f"calibrate {case['id']} {route}: {result['ttft_seconds']:.4f}s "
                  f"fact={result['gold_pass']}", flush=True)
    state["calibration"] = calibration
    state["query_server_pid"] = pid
    save(state, STATE)


def run(cases, endpoint, metrics_url, storage_pressure, args):
    state = json.loads(STATE.read_text())
    if check_server() != state.get("query_server_pid") or "calibration" not in state:
        raise RuntimeError("Calibrate with the restarted vLLM server first")
    if state.get("rows"):
        raise RuntimeError("Routed run already started")
    agent = load_laya()
    rows = []
    for case in cases:
        tokens = matched_tokens(case)
        costs = {mode: round(estimate(state["calibration"], mode, tokens), 4)
                 for mode in ("fresh", "fp8", "tq4")}
        costs["matched_tokens"] = tokens
        costs["available"] = {mode: disk_usage(path)["files"] > 0 for mode, path in L2.items()}
        gpu = gpu_utilization()
        laya_result = None
        laya_seconds = 0.0
        laya_error = None
        if costs["available"]["fp8"] and tokens and costs["fp8"] <= args.fp8_margin * costs["fresh"]:
            try:
                laya_result, laya_seconds = judge(agent, case["question"], tokens)
            except Exception as exc:
                laya_error = f"{type(exc).__name__}: {exc}"
        route, reason = decide(costs, laya_result, storage_pressure, args)
        result = request(case, route, endpoint, metrics_url)
        row = {
            "id": case["id"], "question": case["question"], "expected": case["expected"],
            "route": route, "reason": reason, "cost_estimates_seconds": costs,
            "laya": laya_result, "laya_seconds": laya_seconds,
            "laya_error": laya_error, "gpu_utilization_percent": gpu,
            "combined_ttft_seconds": round(laya_seconds + result["ttft_seconds"], 4),
            **result,
        }
        rows.append(row)
        state["rows"] = rows
        save(state, STATE)
        print(f"{case['id']} {route} L2={result['l2_hit_tokens']} "
              f"TTFT={result['ttft_seconds']:.4f}s gold={result['gold_pass']}", flush=True)


def evaluate():
    agent = load_laya()
    rows = []
    for question, expected in SEMANTIC_PROBES:
        output, seconds = judge(agent, question, 4096)
        probabilities = output["answers"]["sensitivity"]["probabilities"]
        predicted = ("low", "medium", "high")[max(range(3), key=lambda i: probabilities[str(i)])]
        rows.append({
            "question": question, "expected": expected, "predicted": predicted,
            "pass": predicted == expected, "laya_seconds": seconds,
            "answers": output["answers"],
        })
    save({"model_revision": MODEL_REVISION, "cases": rows,
          "correct": sum(row["pass"] for row in rows)}, EVALUATION)
    print(json.dumps({"correct": sum(row["pass"] for row in rows),
                      "total": len(rows), "results": rows}, indent=2))


def report(benchmark):
    state = json.loads(STATE.read_text())
    rows = state.get("rows", [])
    if len(rows) != len(benchmark["cases"]):
        raise RuntimeError(f"Only {len(rows)} of {len(benchmark['cases'])} cases completed")
    baseline = {
        mode: {
            "fact_checks": sum(passed(result["answer"], case["expected"])
                               for result, case in zip(benchmark["runs"][mode]["query"],
                                                       benchmark["cases"])),
            "median_ttft_seconds": statistics.median(
                item["ttft_seconds"] for item in benchmark["runs"][mode]["query"]
            ),
        }
        for mode in ("fresh", "fp8", "turbo")
    }
    summary = {
        "cases": len(rows),
        "route_counts": {route: sum(row["route"] == route for row in rows)
                         for route in ("fresh", "fp8", "tq4")},
        "fact_checks": sum(row["gold_pass"] for row in rows),
        "laya_calls": sum(row["laya_seconds"] > 0 for row in rows),
        "median_laya_seconds": statistics.median(row["laya_seconds"] for row in rows),
        "median_laya_seconds_when_called": statistics.median(
            row["laya_seconds"] for row in rows if row["laya_seconds"] > 0
        ) if any(row["laya_seconds"] > 0 for row in rows) else None,
        "median_vllm_ttft_seconds": statistics.median(row["ttft_seconds"] for row in rows),
        "median_combined_ttft_seconds": statistics.median(
            row["combined_ttft_seconds"] for row in rows
        ),
        "seed_l2_bytes": {mode: state["seed_l2"][mode]["bytes"] for mode in SALTS},
        "final_l2_bytes": {mode: disk_usage(path)["bytes"] for mode, path in L2.items()},
        "historical_baselines": baseline,
        "timing_note": (
            "Exploratory: prior runs had different server settings and run order. "
            "Combined TTFT sums LAYA and vLLM first-token times; Python routing overhead "
            "outside the model call is excluded."
        ),
    }
    save(summary, SUMMARY)
    print(json.dumps(summary, indent=2))


def self_test():
    args = argparse.Namespace(fp8_margin=1.25, tq4_margin=3.0,
                              high_probability=0.5, multi_probability=0.6,
                              low_probability=0.38, lookup_probability=0.25)
    costs = {"matched_tokens": 4096, "available": {"fp8": True, "tq4": True},
             "fresh": 1.0, "fp8": 0.8, "tq4": 2.0}
    normal = {"answers": {"sensitivity": {"probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}},
                          "multi": {"noul": 0.1},
                          "task": {"probabilities": {"lookup": 0.1}}}}
    assert decide(costs, normal, False, args)[0] == "fp8"
    normal["answers"]["sensitivity"]["probabilities"] = {"0": 0.05, "1": 0.05, "2": 0.9}
    assert decide(costs, normal, False, args)[0] == "fresh"
    normal["answers"]["sensitivity"]["probabilities"] = {"0": 0.8, "1": 0.15, "2": 0.05}
    normal["answers"]["task"]["probabilities"]["lookup"] = 0.9
    assert decide(costs, normal, True, args)[0] == "tq4"
    assert decide(costs, None, False, args)[0] == "fresh"
    print("LAYA policy self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("self-test", "evaluate", "seed", "calibrate", "run", "report"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics", default="http://127.0.0.1:8080/metrics")
    parser.add_argument("--storage-pressure", action="store_true")
    parser.add_argument("--fp8-margin", type=float, default=1.25)
    parser.add_argument("--tq4-margin", type=float, default=3.0)
    parser.add_argument("--high-probability", type=float, default=0.5)
    parser.add_argument("--multi-probability", type=float, default=0.6)
    parser.add_argument("--low-probability", type=float, default=0.38)
    parser.add_argument("--lookup-probability", type=float, default=0.25)
    args = parser.parse_args()
    if args.step == "self-test":
        self_test()
    elif args.step == "evaluate":
        evaluate()
    else:
        benchmark = load_cases()
        cases = benchmark["cases"]
        if args.step == "seed":
            seed(cases, args.endpoint, args.metrics)
        elif args.step == "calibrate":
            calibrate(args.endpoint, args.metrics)
        elif args.step == "run":
            run(cases, args.endpoint, args.metrics, args.storage_pressure, args)
        else:
            report(benchmark)


if __name__ == "__main__":
    main()
