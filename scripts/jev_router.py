"""Route second-turn queries to FP8 L2 reuse or salted fresh prefill."""

import argparse
import json
import re
import secrets
import statistics
import time
from pathlib import Path

from benchmark import check_prompt, messages
from compressed_reuse import disk_usage, metrics, passed, save, server_info
from two_turn import MODEL, chat

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "results" / "benchmark.json"
STATE = ROOT / "results" / "jev_router.json"
SUMMARY = ROOT / "results" / "jev_router_summary.json"
L2 = ROOT / "results" / "router_l2" / "fp8"
SENSITIVE = re.compile(r"\b(exact|verbatim|quote|wording|precise|compare|contrast|difference|differences|both)\b")
SECTION_GROUP = re.compile(r"\bsections?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)*)")
LOOKUP = re.compile(r"^(what|which|who|where|when|how many)\b")


def decide(question):
    text = question.casefold().strip()
    marker = SENSITIVE.search(text)
    if marker:
        return "fresh", f"sensitive wording: {marker.group()}"
    sections = [number for group in SECTION_GROUP.findall(text) for number in re.findall(r"\d+", group)]
    if len(set(sections)) > 1:
        return "fresh", "multiple section references"
    if len(sections) == 1 and LOOKUP.match(text):
        return "fp8", "single-section lookup"
    return "fresh", "unrecognized question"


def load_cases():
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    if benchmark["model"] != MODEL or len(benchmark["cases"]) != 27:
        raise RuntimeError("Expected the saved 27-case Llama benchmark")
    return benchmark


def check_server():
    pid, connected = server_info()
    if not connected:
        raise RuntimeError("Start vLLM with the LMCache connector")
    args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    config = json.loads(args[args.index(b"--kv-transfer-config") + 1])
    if (config.get("kv_connector") != "JevLMCacheMPConnector"
            or config.get("kv_connector_module_path") != "jev_connector"):
        raise RuntimeError("Start vLLM with the Jev connector to discard salted KV")
    return pid


def wait_for_seed(before, expected, case_id, metrics_url):
    deadline = time.monotonic() + 90
    while True:
        current = metrics(metrics_url)
        submitted = round(current["submitted"] - before["submitted"])
        completed = round(current["completed"] - before["completed"])
        if submitted == completed == expected and current["l1_bytes"] < 64 * 1024 * 1024:
            return completed
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"{case_id}: L2 store incomplete: {completed}/{submitted}, "
                f"expected {expected}, L1={current['l1_bytes']}"
            )
        time.sleep(0.5)


def wait_for_idle(metrics_url):
    deadline = time.monotonic() + 30
    while True:
        current = metrics(metrics_url)
        if round(current["submitted"]) == round(current["completed"]):
            return current
        if time.monotonic() >= deadline:
            raise RuntimeError("LMCache L2 writes did not finish")
        time.sleep(0.2)


def request(case, route, endpoint, metrics_url):
    before = wait_for_idle(metrics_url)
    before_disk = disk_usage(L2) if route == "fresh" else None
    salt = secrets.token_urlsafe(32) if route == "fresh" else None
    result = chat(endpoint, messages(case, True), cache_salt=salt)
    check_prompt(result, case, True)
    after = metrics(metrics_url)
    result["l1_hit_tokens"] = round(after["l1"] - before["l1"])
    result["l2_hit_tokens"] = round(after["l2"] - before["l2"])
    result["l2_store_submitted_chunks"] = round(after["submitted"] - before["submitted"])
    result["used_cache_salt"] = salt is not None
    result["gold_pass"] = passed(result["answer"], case["expected"])
    if route == "fp8" and (result["l2_hit_tokens"] <= 0 or result["l1_hit_tokens"] != 0):
        raise RuntimeError(f"{case['id']}: FP8 route did not retrieve solely from L2: {result}")
    if route == "fresh":
        # Give asynchronous metrics or disk writes a moment to surface before
        # claiming that this request left no persistent KV behind.
        time.sleep(0.2)
        settled = wait_for_idle(metrics_url)
        result["l2_store_submitted_chunks"] = round(settled["submitted"] - before["submitted"])
        result["l2_data_bytes_added"] = disk_usage(L2)["bytes"] - before_disk["bytes"]
        if (result["l2_hit_tokens"] != 0 or result["l1_hit_tokens"] != 0
                or result["l2_store_submitted_chunks"] != 0 or result["l2_data_bytes_added"] != 0):
            raise RuntimeError(f"{case['id']}: fresh route reused or stored KV: {result}")
    return result


def seed(cases, endpoint, metrics_url):
    pid = check_server()
    if STATE.exists():
        raise RuntimeError(f"Router state already exists: {STATE}")
    if L2.exists() and any(L2.iterdir()):
        raise RuntimeError(f"Use an empty, isolated FP8 L2 directory: {L2}")
    L2.mkdir(parents=True, exist_ok=True)
    before = metrics(metrics_url)
    expected = 0
    seeds = []
    for case in cases:
        result = chat(endpoint, messages(case, False))
        check_prompt(result, case, False)
        seeds.append({"id": case["id"], "answer": result["answer"], "prompt_tokens": result["prompt_tokens"]})
        expected += len(case["seed_token_ids"]) // 256
        wait_for_seed(before, expected, case["id"], metrics_url)
        print(f"seed {case['id']} ({expected} chunks total)", flush=True)
    disk = disk_usage(L2)
    if not disk["files"]:
        raise RuntimeError("No FP8 data files were written to the router L2 directory")
    save({"model": MODEL, "seed_server_pid": pid, "seeded": seeds, "seed_chunks": expected, "seed_l2": disk}, STATE)
    print(json.dumps({"seeded": len(seeds), "chunks": expected, "l2": disk}, indent=2))


def run(cases, endpoint, metrics_url):
    pid = check_server()
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state["model"] != MODEL or len(state["seeded"]) != len(cases):
        raise RuntimeError("Seed state does not match the benchmark")
    if pid == state["seed_server_pid"]:
        raise RuntimeError("Restart vLLM after seeding, keeping LMCache running")
    if state.get("rows") or state.get("preflight"):
        raise RuntimeError("Router run already started; use a fresh isolated seed to repeat")
    # Preflight on a seeded prompt; its unique salt must prevent an L2 hit.
    first = cases[0]
    state["preflight"] = {"id": first["id"], **request(first, "fresh", endpoint, metrics_url)}
    state["query_server_pid"] = pid
    state["rows"] = []
    save(state, STATE)
    print(f"salt preflight {first['id']}: L2={state['preflight']['l2_hit_tokens']}", flush=True)
    for case in cases:
        route, reason = decide(case["question"])
        result = request(case, route, endpoint, metrics_url)
        row = {"id": case["id"], "question": case["question"], "route": route,
               "reason": reason, "expected": case["expected"], **result}
        state["rows"].append(row)
        save(state, STATE)
        print(f"{case['id']} {route} L2={result['l2_hit_tokens']} "
              f"TTFT={result['ttft_seconds']:.4f}s gold={result['gold_pass']}", flush=True)


def report(benchmark):
    state = json.loads(STATE.read_text(encoding="utf-8"))
    rows = state.get("rows", [])
    if len(rows) != len(benchmark["cases"]):
        raise RuntimeError(f"Only {len(rows)} of {len(benchmark['cases'])} routed queries completed")
    baseline = {}
    for mode in ("fresh", "fp8"):
        prior = benchmark["runs"].get(mode, {}).get("query", [])
        if len(prior) == len(rows):
            baseline[mode] = {
                "fact_checks": sum(passed(result["answer"], case["expected"])
                                   for result, case in zip(prior, benchmark["cases"])),
                "median_ttft_seconds": round(statistics.median(r["ttft_seconds"] for r in prior), 4),
            }
    summary = {
        "cases": len(rows),
        "route_counts": {mode: sum(r["route"] == mode for r in rows) for mode in ("fp8", "fresh")},
        "fact_checks": sum(r["gold_pass"] for r in rows),
        "median_ttft_seconds": round(statistics.median(r["ttft_seconds"] for r in rows), 4),
        "seed_l2_bytes": state["seed_l2"]["bytes"],
        "final_l2_bytes": disk_usage(L2)["bytes"],
        "fresh_store_submitted_chunks": sum(r["l2_store_submitted_chunks"] for r in rows if r["route"] == "fresh"),
        "fresh_l2_data_bytes_added": sum(r["l2_data_bytes_added"] for r in rows if r["route"] == "fresh"),
        "fp8_store_submitted_chunks": sum(r["l2_store_submitted_chunks"] for r in rows if r["route"] == "fp8"),
        "historical_baselines": baseline,
        "timing_note": "Exploratory: historical baselines used different server configurations and run order.",
    }
    save(summary, SUMMARY)
    print(json.dumps(summary, indent=2))


def self_test(cases):
    assert decide("Which city is listed in section 5?")[0] == "fp8"
    assert decide("What exact phrase appears in section 13?")[0] == "fresh"
    assert decide("Compare sections 2 and 8.")[0] == "fresh"
    assert decide("What happened in sections 2 and 8?")[0] == "fresh"
    assert decide("What happened?")[0] == "fresh"
    counts = {mode: sum(decide(case["question"])[0] == mode for case in cases) for mode in ("fp8", "fresh")}
    assert counts == {"fp8": 9, "fresh": 18}, counts
    print(f"Router rules passed: {counts}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("self-test", "seed", "run", "report"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics", default="http://127.0.0.1:8080/metrics")
    args = parser.parse_args()
    benchmark = load_cases()
    cases = benchmark["cases"]
    if args.step == "self-test":
        self_test(cases)
    elif args.step == "seed":
        seed(cases, args.endpoint, args.metrics)
    elif args.step == "run":
        run(cases, args.endpoint, args.metrics)
    else:
        report(benchmark)


if __name__ == "__main__":
    main()
