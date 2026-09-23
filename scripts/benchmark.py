import argparse
import json
import statistics
import time
from pathlib import Path

from compressed_reuse import disk_usage, metrics, passed, save, server_info
from two_turn import MODEL, SYSTEM, chat

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "benchmark.json"
SUMMARY = ROOT / "results" / "benchmark_summary.json"
MODES = ("fresh", "raw", "fp8", "turbo")
LENGTHS = (2048, 4096, 8192)
KINDS = ("easy", "exact", "compare")
REPEATS = 3
PHRASES = ("copper gate", "violet lantern", "amber bridge")
CITIES = ("Reno", "Tulsa", "Madison")
COMMITMENTS = (
    ("publish experimental methods", "keep customer data private"),
    ("release safety reports", "protect patient records"),
    ("share audit results", "withhold employee identities"),
)
COMPARE_FACTS = (
    ("experimental methods", "customer data"),
    ("safety reports", "patient records"),
    ("audit results", "employee identities"),
)


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


def make_case(length, kind, repeat, tokenizer):
    case_id = f"{length}-{kind}-{repeat + 1}"
    company = f"Harbor Research {case_id}"
    founder = f"Alex Morgan {repeat + 1}"
    facts = {
        2: f"Founder {founder} said the company would {COMMITMENTS[repeat][0]}.",
        5: f"The headquarters of {company} is in {CITIES[repeat]}.",
        8: f"Founder {founder} said the company would {COMMITMENTS[repeat][1]}.",
        13: f"Founder {founder} called the privacy boundary the {PHRASES[repeat]}.",
        14: "A separate team used the label silver path for its equipment inventory.",
    }
    if kind == "easy":
        question, expected = "Which city is the headquarters in, according to section 5?", [CITIES[repeat]]
    elif kind == "exact":
        question, expected = "What exact two-word name did the founder give the privacy boundary in section 13?", [PHRASES[repeat]]
    else:
        question, expected = "Compare the founder's commitments in sections 2 and 8.", list(COMPARE_FACTS[repeat])
    case = {
        "id": case_id,
        "length": length,
        "kind": kind,
        "repeat": repeat + 1,
        "company": company,
        "founder": founder,
        "question": question,
        "expected": expected,
    }

    count = max(20, length // 36)
    for _ in range(8):
        sections = [f"Document ID {case_id}. Company: {company}. Founder: {founder}."]
        for number in range(1, count + 1):
            body = facts.get(number)
            if body is None:
                body = (
                    f"The {company} team recorded ordinary observations about its "
                    "research process, meeting schedule, equipment checks, and internal "
                    "review. This section adds background and no new founder commitment."
                )
            sections.append(f"Section {number}: {body}")
        case["document"] = "\n".join(sections)
        case["seed_token_ids"] = tokenizer.apply_chat_template(
            messages(case, False), tokenize=True, add_generation_prompt=True
        )["input_ids"]
        case["query_token_ids"] = tokenizer.apply_chat_template(
            messages(case, True), tokenize=True, add_generation_prompt=True
        )["input_ids"]
        delta = length - len(case["query_token_ids"])
        if abs(delta) <= 80:
            break
        count = max(20, count + (max(1, round(delta / 39)) if delta > 0 else min(-1, round(delta / 39))))
    if abs(length - len(case["query_token_ids"])) > 100 or len(case["query_token_ids"]) > 9216:
        raise RuntimeError(f"{case_id}: query length {len(case['query_token_ids'])} misses target {length}")
    case["document_token_ids"] = tokenizer.encode(case["document"], add_special_tokens=False)
    return case


def check_prompt(result, case, second_turn):
    key = "query_token_ids" if second_turn else "seed_token_ids"
    if result["prompt_tokens"] != len(case[key]):
        raise RuntimeError(f"{case['id']}: vLLM prompt count differs from saved token IDs")


def run_queries(state, mode, endpoint, metrics_url):
    results = []
    for case in state["cases"]:
        before = metrics(metrics_url) if mode != "fresh" else None
        result = chat(endpoint, messages(case, True))
        check_prompt(result, case, True)
        if before is not None:
            after = metrics(metrics_url)
            result["l1_hit_tokens"] = round(after["l1"] - before["l1"])
            result["l2_hit_tokens"] = round(after["l2"] - before["l2"])
            if result["l2_hit_tokens"] <= 0 or result["l1_hit_tokens"] != 0:
                raise RuntimeError(f"{mode}/{case['id']}: expected L2-only retrieval: {result}")
        result["gold_pass"] = passed(result["answer"], case["expected"])
        results.append(result)
        print(f"{mode} {case['id']} {result['ttft_seconds']:.4f}s gold={result['gold_pass']}"
              + (f" L2={result['l2_hit_tokens']}" if before is not None else ""), flush=True)
    return results


def report(state):
    if any(len(state["runs"].get(mode, {}).get("query", [])) != len(state["cases"]) for mode in MODES):
        raise RuntimeError("All four modes must complete before reporting.")
    runs = state["runs"]
    summary = {
        "model": MODEL,
        "cases": len(state["cases"]),
        "storage_bytes": {mode: runs[mode]["disk"]["bytes"] for mode in MODES[1:]},
        "accuracy": {},
        "median_ttft_seconds": {},
        "cells": [],
    }
    for mode in MODES:
        rows = runs[mode]["query"]
        summary["accuracy"][mode] = sum(row["gold_pass"] for row in rows)
        summary["median_ttft_seconds"][mode] = round(statistics.median(row["ttft_seconds"] for row in rows), 4)
    for length in LENGTHS:
        for kind in KINDS:
            indices = [i for i, case in enumerate(state["cases"]) if case["length"] == length and case["kind"] == kind]
            summary["cells"].append({
                "length": length,
                "kind": kind,
                "accuracy": {mode: sum(runs[mode]["query"][i]["gold_pass"] for i in indices) for mode in MODES},
                "median_ttft_seconds": {
                    mode: round(statistics.median(runs[mode]["query"][i]["ttft_seconds"] for i in indices), 4)
                    for mode in MODES
                },
                "median_l2_hit_tokens": {
                    mode: statistics.median(runs[mode]["query"][i]["l2_hit_tokens"] for i in indices)
                    for mode in MODES[1:]
                },
            })
    summary["compressed_answer_changes"] = {
        mode: sum(
            runs[mode]["query"][i]["answer"] != runs["raw"]["query"][i]["answer"]
            for i in range(len(state["cases"]))
        )
        for mode in ("fp8", "turbo")
    }
    save(summary, SUMMARY)
    print(json.dumps(summary, indent=2))


def regrade(state):
    """Score saved answers against the current fact checks."""
    for case in state["cases"]:
        if case["kind"] == "compare":
            case["expected"] = list(COMPARE_FACTS[case["repeat"] - 1])
    for run in state["runs"].values():
        for case, result in zip(state["cases"], run.get("query", [])):
            result["gold_pass"] = passed(result["answer"], case["expected"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("init", "fresh", "seed", "query", "report"))
    parser.add_argument("--mode", choices=MODES[1:])
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics", default="http://127.0.0.1:8080/metrics")
    args = parser.parse_args()

    if args.step == "init":
        from transformers import AutoTokenizer

        if STATE.exists():
            raise RuntimeError(f"Results already exist: {STATE}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        cases = [make_case(length, kind, repeat, tokenizer)
                 for length in LENGTHS for kind in KINDS for repeat in range(REPEATS)]
        save({"model": MODEL, "cases": cases, "runs": {}}, STATE)
        print(json.dumps({"cases": len(cases), "prompt_ranges": {
            length: [min(len(c["query_token_ids"]) for c in cases if c["length"] == length),
                     max(len(c["query_token_ids"]) for c in cases if c["length"] == length)]
            for length in LENGTHS
        }}))
        return

    state = json.loads(STATE.read_text(encoding="utf-8"))
    if args.step == "report":
        regrade(state)
        save(state, STATE)
        report(state)
        return
    if args.step in ("seed", "query") and not args.mode:
        parser.error("--mode is required for seed and query")
    mode = "fresh" if args.step == "fresh" else args.mode
    pid, connected = server_info()
    if connected == (mode == "fresh"):
        raise RuntimeError("Fresh needs vLLM without LMCache; cached modes need the connector.")
    run = state["runs"].setdefault(mode, {})

    if args.step == "seed":
        if "seed" in run:
            raise RuntimeError(f"{mode} already seeded")
        l2_path = ROOT / "results" / "benchmark_l2" / mode
        if l2_path.exists() and any(l2_path.iterdir()):
            raise RuntimeError(f"Use an empty L2 directory: {l2_path}")
        before = metrics(args.metrics)
        seeds = []
        for case in state["cases"]:
            result = chat(args.endpoint, messages(case, False))
            check_prompt(result, case, False)
            seeds.append(result)
            print(f"{mode} seed {case['id']}", flush=True)
            expected = sum(len(item["seed_token_ids"]) // 256 for item in state["cases"][:len(seeds)])
            deadline = time.monotonic() + 90
            while True:
                current = metrics(args.metrics)
                submitted = current["submitted"] - before["submitted"]
                completed = current["completed"] - before["completed"]
                if submitted == completed == expected and current["l1_bytes"] < 64 * 1024 * 1024:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"{mode}/{case['id']}: L2 store incomplete: "
                        f"{completed}/{submitted}, expected {expected}, L1={current['l1_bytes']}"
                    )
                time.sleep(0.5)
        run.update(seed=seeds, seed_server_pid=pid, disk=disk_usage(l2_path))
        if not run["disk"]["files"]:
            raise RuntimeError(f"{mode}: no L2 data files")
        save(state, STATE)
        print(json.dumps({"mode": mode, "stored_chunks": int(completed), "disk": run["disk"]}))
        return

    if "query" in run:
        raise RuntimeError(f"{mode} query already complete")
    if mode != "fresh" and ("seed" not in run or pid == run["seed_server_pid"]):
        raise RuntimeError(f"{mode}: seed, then restart vLLM before query")
    run["query"] = run_queries(state, mode, args.endpoint, args.metrics)
    run["query_server_pid"] = pid
    save(state, STATE)


if __name__ == "__main__":
    main()
