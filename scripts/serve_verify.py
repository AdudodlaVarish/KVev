"""Verify hot L1, cold FP8 L2, and store-free fresh routes across a vLLM restart."""

import argparse
import json
import secrets
import time
from pathlib import Path
from urllib.request import Request, urlopen

from compressed_reuse import disk_usage, save
from jev_router import check_server, wait_for_idle
from serve_benchmark import L2, STATE as BENCHMARK, first_messages, followup_messages
from two_turn import chat

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "results" / "serve_verify.json"
PROXY = "http://127.0.0.1:8001"
METRICS = "http://127.0.0.1:8080/metrics"


def delta(before, after):
    return {name: round(after[name] - before[name]) for name in ("l1", "l2", "submitted")}


def main(step):
    pid = check_server()
    if step == "seed":
        if STATE.exists():
            raise RuntimeError(f"Verification state exists: {STATE}")
        case = json.loads(BENCHMARK.read_text())["hot"][0]
        case["article"] += "\nVerification ID: " + secrets.token_hex(16)
        before = wait_for_idle(METRICS)
        before_bytes = disk_usage(L2)["bytes"]
        result = chat(PROXY, first_messages(case), max_tokens=16)
        deadline = time.monotonic() + 90
        while True:
            current = wait_for_idle(METRICS)
            disk = disk_usage(L2)
            if (current["l1_bytes"] > 0 and disk["bytes"] > before_bytes
                    and current["submitted"] > before["submitted"]):
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Seed was not retained in L1 and written to L2")
            time.sleep(0.5)
        state = {"seed_pid": pid, "case": case, "seed": result,
                 "seed_l2_bytes": disk["bytes"]}
    else:
        state = json.loads(STATE.read_text())
        case = state["case"]
        if step == "cold":
            if pid == state["seed_pid"]:
                raise RuntimeError("Restart vLLM before cold verification")
            request = Request("http://127.0.0.1:8080/cache/clear", data=b"", method="POST")
            with urlopen(request, timeout=10) as response:
                if response.status != 200:
                    raise RuntimeError("Could not clear L1 before L2 test")
        before = wait_for_idle(METRICS)
        before_bytes = disk_usage(L2)["bytes"]
        if step == "fresh":
            prompt = first_messages(case) + [
                {"role": "assistant", "content": "ABCD"[case["seed_question"]["gold_label"] - 1]},
                {"role": "user", "content": "Quote the article's exact opening sentence verbatim."},
            ]
        else:
            prompt = followup_messages(case, case["questions"][0])
        result = chat(PROXY, prompt, max_tokens=16)
        time.sleep(0.2)
        after = wait_for_idle(METRICS)
        changes = delta(before, after)
        result.update(changes)
        result["l2_bytes_added"] = disk_usage(L2)["bytes"] - before_bytes
        if step == "hot" and (changes["l1"] <= 0 or changes["l2"] != 0):
            raise RuntimeError(f"Hot request did not use L1: {result}")
        if step == "cold" and (changes["l1"] != 0 or changes["l2"] <= 0):
            raise RuntimeError(f"Cold request did not use FP8 L2: {result}")
        if step == "fresh" and (any(changes.values()) or result["l2_bytes_added"]):
            raise RuntimeError(f"Fresh request hit or wrote cache: {result}")
        state[step] = result
    save(state, STATE)
    print(json.dumps({"step": step, "result": state[step], "state": str(STATE)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("seed", "hot", "cold", "fresh"))
    main(parser.parse_args().step)
