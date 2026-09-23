import argparse
import json
import re
import statistics
import time
from pathlib import Path
from urllib.request import urlopen

from two_turn import MODEL, SYSTEM, chat

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "compressed_reuse.json"
L2_ROOT = ROOT / "results" / "l2"
METRICS = {
    "l1": "lmcache_mp_lookup_hit_l1_tokens_total",
    "l2": "lmcache_mp_lookup_hit_l2_tokens_total",
    "submitted": "lmcache_mp_l2_store_submitted_objects_chunks_total",
    "completed": "lmcache_mp_l2_store_completed_objects_chunks_total",
    "l1_bytes": "lmcache_mp_l1_memory_usage_bytes",
}
CASES = [
    {
        "id": "phrase",
        "company": "Orion Materials",
        "founder": "Hana Vale",
        "facts": {8: "Founder Hana Vale called the privacy boundary the copper gate."},
        "question": "What exact two-word name did the founder give the privacy boundary in section 8?",
        "expected": ["copper gate"],
    },
    {
        "id": "number",
        "company": "Lumen Health",
        "founder": "Idris Chen",
        "facts": {13: "Founder Idris Chen pledged $4.2 million to clinic upgrades by 2025."},
        "question": "How much did the founder pledge in section 13, and by what year?",
        "expected": ["$4.2 million", "2025"],
    },
    {
        "id": "comparison",
        "company": "Meridian Labs",
        "founder": "Mira Solis",
        "facts": {
            2: "Founder Mira Solis said the company would publish its experimental methods.",
            8: "Founder Mira Solis said the company would keep customer data private.",
        },
        "question": "Compare the founder's commitments in sections 2 and 8.",
        "expected": ["experimental methods", "customer data"],
    },
    {
        "id": "exception",
        "company": "Cedar Robotics",
        "founder": "Nia Okafor",
        "facts": {13: "Founder Nia Okafor allowed sharing customer data only after written consent."},
        "question": "What condition did the founder set for sharing customer data in section 13?",
        "expected": ["written consent"],
    },
]


def document(case):
    sections = [f"Document for {case['company']}. Founder: {case['founder']}."]
    for number in range(1, 81):
        body = case["facts"].get(number)
        if body is None:
            body = (
                f"The {case['company']} team recorded routine observations about its "
                "research process, meeting schedule, equipment checks, and internal "
                "review. This section adds background and no new founder commitment."
            )
        sections.append(f"Section {number}: {body}")
    return "\n".join(sections)


def messages(case, second_turn):
    first = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Document:\n{case['document']}\n\nQuestion: Who founded {case['company']}?"},
    ]
    if not second_turn:
        return first
    return first + [
        {"role": "assistant", "content": f"{case['founder']} founded {case['company']}."},
        {"role": "user", "content": case["question"]},
    ]


def metrics(url):
    with urlopen(url, timeout=10) as response:
        body = response.read().decode("utf-8")
    result = {name: 0 for name in METRICS}
    for line in body.splitlines():
        for name, metric in METRICS.items():
            if line.startswith(metric) and line[len(metric) : len(metric) + 1] in ("{", " "):
                result[name] += float(line.rsplit(None, 1)[-1])
    return result


def server_info():
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().split(b"\0")
            if len(args) > 2 and args[1].endswith(b"/vllm") and args[2] == b"serve":
                return int(path.parent.name), b"--kv-transfer-config" in args
        except (OSError, PermissionError):
            pass
    raise RuntimeError("No local vLLM serve process found.")


def disk_usage(path):
    files = list(path.rglob("*.data"))
    return {"files": len(files), "bytes": sum(file.stat().st_size for file in files)}


def save(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def passed(answer, expected):
    normalized = re.sub(r"\s+", " ", answer).casefold()
    return all(term.casefold() in normalized for term in expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("init", "fresh", "raw-seed", "raw-query", "fp8-seed", "fp8-query", "report"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics", default="http://127.0.0.1:8080/metrics")
    parser.add_argument("--state", type=Path, default=STATE)
    args = parser.parse_args()

    if args.step == "init":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        cases = []
        for source in CASES:
            case = {key: value for key, value in source.items() if key != "facts"}
            case["document"] = document(source)
            case["document_token_ids"] = tokenizer.encode(case["document"], add_special_tokens=False)
            case["seed_token_ids"] = tokenizer.apply_chat_template(messages(case, False), tokenize=True, add_generation_prompt=True)["input_ids"]
            case["query_token_ids"] = tokenizer.apply_chat_template(messages(case, True), tokenize=True, add_generation_prompt=True)["input_ids"]
            if not 2000 <= len(case["document_token_ids"]) <= 4000 or len(case["query_token_ids"]) > 4096:
                raise RuntimeError(f"Prompt length outside test range: {case['id']}")
            cases.append(case)
        state = {"model": MODEL, "cases": cases, "runs": {}}
        save(state, args.state)
        print(json.dumps({"cases": [(c["id"], len(c["document_token_ids"]), len(c["query_token_ids"])) for c in cases], "state": str(args.state)}))
        return

    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state["model"] != MODEL:
        raise RuntimeError("Saved state uses a different model.")
    step = args.step
    if step == "report":
        runs = state["runs"]
        if set(runs) != {"fresh", "raw", "fp8"} or not all("query" in runs[name] for name in runs):
            raise RuntimeError("Complete all three conditions before reporting.")
        if runs["fp8"]["disk"]["bytes"] >= runs["raw"]["disk"]["bytes"]:
            raise RuntimeError("FP8 L2 is not smaller than raw L2.")
        summary = {
            "seed_l2_bytes": {name: runs[name]["disk"]["bytes"] for name in ("raw", "fp8")},
            "fp8_size_ratio": round(runs["fp8"]["disk"]["bytes"] / runs["raw"]["disk"]["bytes"], 3),
            "correct_cases": {
                name: sum(passed(result["answer"], case["expected"]) for result, case in zip(runs[name]["query"], state["cases"]))
                for name in ("fresh", "raw", "fp8")
            },
            "median_ttft_seconds": {
                name: round(statistics.median(result["ttft_seconds"] for result in runs[name]["query"]), 4)
                for name in ("fresh", "raw", "fp8")
            },
            "cases": [],
        }
        for index, case in enumerate(state["cases"]):
            summary["cases"].append({
                "id": case["id"],
                "gold": case["expected"],
                "answer_changed": {
                    "raw_vs_fresh": runs["raw"]["query"][index]["answer"] != runs["fresh"]["query"][index]["answer"],
                    "fp8_vs_raw": runs["fp8"]["query"][index]["answer"] != runs["raw"]["query"][index]["answer"],
                },
                "results": {
                    name: {
                        "correct": passed(runs[name]["query"][index]["answer"], case["expected"]),
                        "answer": runs[name]["query"][index]["answer"],
                        "ttft_seconds": runs[name]["query"][index]["ttft_seconds"],
                    }
                    for name in ("fresh", "raw", "fp8")
                },
            })
        print(json.dumps(summary, indent=2))
        return

    condition, _, action = step.partition("-")
    if step == "fresh":
        condition, action = "fresh", "query"
    run = state["runs"].setdefault(condition, {})
    pid, connected = server_info()
    if connected == (condition == "fresh"):
        raise RuntimeError("Fresh runs need vLLM without LMCache; L2 runs need the connector.")
    if action == "seed":
        if "seed" in run:
            raise RuntimeError(f"{condition} already seeded; run init to start over.")
        l2_path = L2_ROOT / condition
        if l2_path.exists() and any(l2_path.iterdir()):
            raise RuntimeError(f"Expected an empty L2 directory: {l2_path}")
        before = metrics(args.metrics)
        run["seed"] = [chat(args.endpoint, messages(case, False)) for case in state["cases"]]
        if any(result["prompt_tokens"] != len(case["seed_token_ids"]) for result, case in zip(run["seed"], state["cases"])):
            raise RuntimeError("Seed prompt token IDs differ from vLLM's prompt count.")
        deadline = time.monotonic() + 120
        while True:
            current = metrics(args.metrics)
            submitted = current["submitted"] - before["submitted"]
            completed = current["completed"] - before["completed"]
            if submitted > 0 and completed >= submitted:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"L2 store did not finish: {completed}/{submitted} objects")
            time.sleep(0.5)
        run["seed_server_pid"] = pid
        run["disk"] = disk_usage(l2_path)
        if not run["disk"]["files"]:
            raise RuntimeError("L2 store finished but no data files were found.")
        save(state, args.state)
        print(json.dumps({"condition": condition, "stored_objects": int(completed), "disk": run["disk"], "server_pid": pid}))
        return

    if condition != "fresh" and ("seed" not in run or pid == run["seed_server_pid"]):
        raise RuntimeError("Seed this condition, then restart vLLM before querying.")
    if "query" in run:
        raise RuntimeError(f"{condition} already queried; run init to start over.")
    results = []
    for case in state["cases"]:
        before = metrics(args.metrics) if condition != "fresh" else None
        result = chat(args.endpoint, messages(case, True))
        if result["prompt_tokens"] != len(case["query_token_ids"]):
            raise RuntimeError(f"Query prompt token IDs differ for {case['id']}.")
        if before is not None:
            after = metrics(args.metrics)
            result["l1_hit_tokens"] = round(after["l1"] - before["l1"])
            result["l2_hit_tokens"] = round(after["l2"] - before["l2"])
            if result["l2_hit_tokens"] <= 0 or result["l1_hit_tokens"] != 0:
                raise RuntimeError(f"{condition}/{case['id']} did not retrieve solely from L2: {result}")
        result["gold_pass"] = passed(result["answer"], case["expected"])
        results.append(result)
        print(json.dumps({"case": case["id"], **result}))
    run["query"] = results
    run["query_server_pid"] = pid
    save(state, args.state)


if __name__ == "__main__":
    main()
