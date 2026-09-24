"""Paired QuALITY workload for direct vLLM and the JevKV proxy."""

import argparse
import hashlib
import json
import random
import re
import statistics
import time
import urllib.request
import zipfile
from pathlib import Path

from compressed_reuse import disk_usage, metrics, save, server_info
from jev_router import check_server, wait_for_idle
from serve_proxy import simple_lookup, token_count
from two_turn import MODEL, chat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "quality" / "QuALITY.v1.0.1.zip"
STATE = ROOT / "results" / "serve_benchmark.json"
SUMMARY = ROOT / "results" / "serve_benchmark_summary.json"
L2 = ROOT / "results" / "serve_l2" / "fp8"
DATA_URL = "https://raw.githubusercontent.com/nyu-mll/quality/main/data/v1.0.1/QuALITY.v1.0.1.zip"
DATA_SHA256 = "58552970c3dc4fe199654579e9f74139becd90fca61ddc2eb0b0d82cbb437341"
SYSTEM = "Use only the article. Answer multiple choice questions with one capital letter.\nArticle:\n"


def question_text(question):
    options = "\n".join(f"{letter}. {answer}"
                        for letter, answer in zip("ABCD", question["options"]))
    return f'{question["question"]}\n{options}\nAnswer with one letter.'


def first_messages(case):
    return [
        {"role": "system", "content": SYSTEM + case["article"]},
        {"role": "user", "content": question_text(case["seed_question"])},
    ]


def followup_messages(case, question):
    return first_messages(case) + [
        {"role": "assistant", "content": "ABCD"[case["seed_question"]["gold_label"] - 1]},
        {"role": "user", "content": question_text(question)},
    ]


def download_data():
    DATA.parent.mkdir(parents=True, exist_ok=True)
    if not DATA.exists():
        urllib.request.urlretrieve(DATA_URL, DATA)
    actual = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if actual != DATA_SHA256:
        raise RuntimeError(f"QuALITY archive checksum mismatch: {actual}")


def prepare():
    if STATE.exists():
        raise RuntimeError(f"Benchmark state exists: {STATE}")
    from transformers import AutoTokenizer

    download_data()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    with zipfile.ZipFile(DATA) as archive:
        rows = [json.loads(line) for line in
                archive.open("QuALITY.v1.0.1.htmlstripped.dev")]
    articles = {}
    for row in rows:
        articles.setdefault(row["article_id"], row)
    candidates = []
    for article_id, row in sorted(articles.items()):
        questions = row["questions"]
        if len(questions) < 4:
            continue
        case = {
            "id": article_id, "article": row["article"],
            "seed_question": questions[0],
        }
        first = first_messages(case)
        seed_tokens = token_count(tokenizer, first, True)
        if not 6000 <= seed_tokens <= 8000:
            continue
        eligible = [question for question in questions[1:]
                    if simple_lookup(question["question"])[0]
                    and token_count(tokenizer, followup_messages(case, question), True) <= 9216]
        if len(eligible) < 3:
            continue
        case["seed_tokens"] = seed_tokens
        case["questions"] = eligible[:3]
        candidates.append(case)
    if len(candidates) < 20:
        raise RuntimeError(f"Only {len(candidates)} eligible QuALITY articles")
    state = {"model": MODEL, "dataset_url": DATA_URL, "dataset_sha256": DATA_SHA256,
             "hot": candidates[:4], "distractors": candidates[4:20], "runs": {}}
    save(state, STATE)
    print(json.dumps({"hot_articles": [c["id"] for c in state["hot"]],
                      "distractors": [c["id"] for c in state["distractors"]],
                      "seed_tokens": [c["seed_tokens"] for c in state["hot"]]}, indent=2))


def choice(answer):
    match = re.search(r"\b([ABCD])\b", answer.upper())
    return match.group(1) if match else None


def run(mode, endpoint, metrics_url):
    state = json.loads(STATE.read_text())
    if mode in state["runs"]:
        raise RuntimeError(f"{mode} already completed")
    _, connected = server_info()
    if connected != (mode == "jevkv"):
        raise RuntimeError("Use vLLM without LMCache for baseline, with Jev connector for jevkv")
    if mode == "jevkv":
        check_server()
        if disk_usage(L2)["files"]:
            raise RuntimeError(f"Use an empty benchmark L2 directory: {L2}")
        with urllib.request.urlopen(endpoint + "/health", timeout=5) as response:
            if json.load(response)["status"] != "ready":
                raise RuntimeError("JevKV proxy is not ready")
    results = {"seed": [], "distractors": [], "query": []}
    for case in state["hot"]:
        result = chat(endpoint, first_messages(case), max_tokens=16)
        results["seed"].append({"id": case["id"], **result})
        print(f"{mode} seed {case['id']}: {result['ttft_seconds']:.4f}s", flush=True)
    if mode == "jevkv":
        deadline = time.monotonic() + 90
        while True:
            current = wait_for_idle(metrics_url)
            if current["l1_bytes"] > 100_000_000 and disk_usage(L2)["bytes"] > 0:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Hot prefixes were not retained in L1 and written to FP8 L2")
            time.sleep(0.5)
    for round_index in range(3):
        distractors = list(state["distractors"])
        random.Random(100 + round_index).shuffle(distractors)
        for case in distractors:
            headers = {"X-JevKV-Cache": "bypass"} if mode == "jevkv" else None
            result = chat(endpoint, first_messages(case), headers=headers, max_tokens=16)
            results["distractors"].append({"round": round_index + 1, "id": case["id"],
                                           "ttft_seconds": result["ttft_seconds"]})
        hot = list(state["hot"])
        random.Random(200 + round_index).shuffle(hot)
        for case in hot:
            question = case["questions"][round_index]
            before = wait_for_idle(metrics_url) if mode == "jevkv" else None
            result = chat(endpoint, followup_messages(case, question), max_tokens=16)
            row = {"round": round_index + 1, "id": case["id"],
                   "question_id": question.get("question_unique_id"),
                   "gold": "ABCD"[question["gold_label"] - 1],
                   "choice": choice(result["answer"]), **result}
            row["correct"] = row["choice"] == row["gold"]
            if before is not None:
                after = wait_for_idle(metrics_url)
                row["l1_hit_tokens"] = round(after["l1"] - before["l1"])
                row["l2_hit_tokens"] = round(after["l2"] - before["l2"])
                if row["l1_hit_tokens"] <= 0:
                    raise RuntimeError(f"Expected hot L1 retrieval: {row}")
            results["query"].append(row)
            state["runs"][mode] = results
            save(state, STATE)
            print(f"{mode} round {round_index + 1} {case['id']}: "
                  f"{row['ttft_seconds']:.4f}s, choice={row['choice']}, "
                  f"correct={row['correct']}"
                  + (f", L1={row['l1_hit_tokens']}" if before is not None else ""),
                  flush=True)
    results["l2_data_bytes"] = disk_usage(L2)["bytes"] if mode == "jevkv" else 0
    state["runs"][mode] = results
    save(state, STATE)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (position - low)


def report():
    state = json.loads(STATE.read_text())
    runs = state["runs"]
    if any(len(runs.get(mode, {}).get("query", [])) != 12 for mode in ("baseline", "jevkv")):
        raise RuntimeError("Both modes need 12 completed follow-ups")
    summary = {"model": MODEL, "dataset": DATA_URL, "hot_articles": 4,
               "distractor_articles": 16, "rounds": 3, "modes": {}}
    for mode in ("baseline", "jevkv"):
        rows = runs[mode]["query"]
        times = [row["ttft_seconds"] for row in rows]
        summary["modes"][mode] = {
            "median_ttft_seconds": round(statistics.median(times), 4),
            "p95_ttft_seconds": round(percentile(times, 0.95), 4),
            "correct": sum(row["correct"] for row in rows),
            "l2_data_bytes": runs[mode]["l2_data_bytes"],
        }
        if mode == "jevkv":
            summary["modes"][mode]["l1_hit_tokens"] = sum(row["l1_hit_tokens"] for row in rows)
            summary["modes"][mode]["l2_hit_tokens"] = sum(row["l2_hit_tokens"] for row in rows)
    baseline = summary["modes"]["baseline"]
    jevkv = summary["modes"]["jevkv"]
    summary["median_ttft_improvement_percent"] = round(
        100 * (1 - jevkv["median_ttft_seconds"] / baseline["median_ttft_seconds"]), 1)
    summary["passes_performance_gate"] = summary["median_ttft_improvement_percent"] >= 10
    summary["passes_quality_gate"] = jevkv["correct"] >= baseline["correct"] - 1
    summary["note"] = "Sequential local runs; paired prompts and order, but different server startups."
    save(summary, SUMMARY)
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "run", "report"))
    parser.add_argument("--mode", choices=("baseline", "jevkv"))
    parser.add_argument("--endpoint")
    parser.add_argument("--metrics", default="http://127.0.0.1:8080/metrics")
    args = parser.parse_args()
    if args.step == "prepare":
        prepare()
    elif args.step == "run":
        if not args.mode:
            parser.error("run requires --mode")
        endpoint = args.endpoint or ("http://127.0.0.1:8001" if args.mode == "jevkv"
                                     else "http://127.0.0.1:8000")
        run(args.mode, endpoint, args.metrics)
    else:
        report()


if __name__ == "__main__":
    main()
