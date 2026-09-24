"""Measure the shortest prefix where local L1 reuse beats salted prefill."""

import json
import secrets
import statistics
import time
from pathlib import Path

from benchmark import messages
from compressed_reuse import metrics, save
from jev_router import check_server, wait_for_idle
from serve_benchmark import STATE as QUALITY_STATE, first_messages
from cache_policies import SALTS
from two_turn import chat

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "serve_calibration.json"
BENCHMARK = ROOT / "results" / "benchmark.json"


def run():
    if OUTPUT.exists():
        raise RuntimeError(f"Calibration exists: {OUTPUT}")
    check_server()
    benchmark = json.loads(BENCHMARK.read_text())
    quality = json.loads(QUALITY_STATE.read_text())
    cases = [case for case in benchmark["cases"] if case["kind"] == "easy"]
    if len(cases) != 9:
        raise RuntimeError("Expected nine saved easy-lookup probes")
    for case in cases:
        chat("http://127.0.0.1:8000", messages(case, False),
             cache_salt=SALTS["fp8"], max_tokens=16)
    deadline = time.monotonic() + 90
    while True:
        current = wait_for_idle("http://127.0.0.1:8080/metrics")
        if current["l1_bytes"] > 100_000_000:
            break
        if time.monotonic() > deadline:
            raise RuntimeError("Probe prefixes were not retained in L1")
        time.sleep(0.5)
    for distractor in quality["distractors"]:
        chat("http://127.0.0.1:8000", first_messages(distractor),
             cache_salt=secrets.token_urlsafe(32), max_tokens=16)
    rows = []
    for index, case in enumerate(cases):
        prompt = messages(case, True)
        before = metrics("http://127.0.0.1:8080/metrics")
        if index % 2:
            fresh = chat("http://127.0.0.1:8000", prompt,
                         cache_salt=secrets.token_urlsafe(32), max_tokens=16)
            cached = chat("http://127.0.0.1:8000", prompt,
                          cache_salt=SALTS["fp8"], max_tokens=16)
        else:
            cached = chat("http://127.0.0.1:8000", prompt,
                          cache_salt=SALTS["fp8"], max_tokens=16)
            fresh = chat("http://127.0.0.1:8000", prompt,
                         cache_salt=secrets.token_urlsafe(32), max_tokens=16)
        after = wait_for_idle("http://127.0.0.1:8080/metrics")
        hit = round(after["l1"] - before["l1"])
        if hit <= 0 or after["l2"] - before["l2"] > 0:
            raise RuntimeError(f"Expected an L1-only probe: {case['id']}, L1={hit}")
        row = {"id": case["id"], "length": case["length"],
               "matched_tokens": hit, "l1_ttft_seconds": cached["ttft_seconds"],
               "fresh_ttft_seconds": fresh["ttft_seconds"]}
        rows.append(row)
        print(json.dumps(row), flush=True)
    medians = {}
    for length in (2048, 4096, 8192):
        subset = [row for row in rows if row["length"] == length]
        medians[length] = {
            "l1": round(statistics.median(row["l1_ttft_seconds"] for row in subset), 4),
            "fresh": round(statistics.median(row["fresh_ttft_seconds"] for row in subset), 4),
        }
    winners = [length for length in medians
               if medians[length]["l1"] <= 0.9 * medians[length]["fresh"]]
    threshold = min(winners) if winners else 4096
    result = {"min_prefix_tokens": threshold, "measured_10_percent_winner": bool(winners),
              "medians": medians, "rows": rows,
              "note": "Threshold is a conservative 4096-token fallback if no probe wins."}
    save(result, OUTPUT)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
