"""Check the serving proxy against the original 27 fixed-fact questions."""

import json
from pathlib import Path

from benchmark import messages
from compressed_reuse import passed, save
from jev_router import load_cases
from two_turn import chat

OUTPUT = Path(__file__).resolve().parents[1] / "results" / "serve_facts.json"


def main():
    cases = load_cases()["cases"]
    rows = []
    for case in cases:
        chat("http://127.0.0.1:8001", messages(case, False), max_tokens=32)
        result = chat("http://127.0.0.1:8001", messages(case, True))
        row = {"id": case["id"], "expected": case["expected"],
               "answer": result["answer"], "pass": passed(result["answer"], case["expected"]),
               "ttft_seconds": result["ttft_seconds"]}
        rows.append(row)
        print(f"{row['id']}: {row['pass']}", flush=True)
    save({"correct": sum(row["pass"] for row in rows), "total": len(rows), "rows": rows},
         OUTPUT)
    if not all(row["pass"] for row in rows):
        raise SystemExit("Serving fact check failed; see results/serve_facts.json")
    print(json.dumps({"correct": len(rows), "total": len(rows)}))


if __name__ == "__main__":
    main()
