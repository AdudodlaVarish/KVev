import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

MODEL = "meta-llama/Llama-3.2-1B-Instruct"
STATE = Path(__file__).resolve().parents[1] / "results" / "two_turn.json"
HIT_METRIC = "lmcache_mp_lookup_hit_tokens_total"
SYSTEM = "Answer the user's question using only the numbered document sections. Be concise."
QUESTION_1 = "Who founded Meridian Labs?"
QUESTION_2 = "Compare the founder's statements in sections 2 and 8."


def document():
    sections = []
    for number in range(1, 81):
        if number == 2:
            body = (
                "Mira Solis founded Meridian Labs in 2017. She said the company "
                "would publish its experimental methods so other researchers "
                "could check the results."
            )
        elif number == 8:
            body = (
                "Founder Mira Solis said Meridian Labs would keep customer data "
                "private, even when sharing its experimental methods. She called "
                "privacy a firm limit on transparency."
            )
        else:
            body = (
                "The team recorded routine observations about its research process, "
                "meeting schedule, equipment checks, and internal review. This "
                "section adds background but makes no claim about the founder."
            )
        sections.append(f"Section {number}: {body}")
    return "\n".join(sections)


def turn_one_messages(doc):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Document:\n{doc}\n\nQuestion: {QUESTION_1}"},
    ]


def turn_two_messages(doc, answer):
    return turn_one_messages(doc) + [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": QUESTION_2},
    ]


def metric_total(url):
    with urlopen(url, timeout=10) as response:
        body = response.read().decode("utf-8")
    total = 0.0
    for line in body.splitlines():
        if line.startswith(HIT_METRIC) and line[len(HIT_METRIC) : len(HIT_METRIC) + 1] in ("{", " "):
            total += float(line.rsplit(None, 1)[-1])
    return total


def chat(endpoint, messages, cache_salt=None):
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 160,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    request = Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    parts = []
    first_token_seconds = None
    usage = None
    with urlopen(request, timeout=180) as response:
        for raw_line in response:
            if not raw_line.startswith(b"data: "):
                continue
            data = raw_line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    if first_token_seconds is None:
                        first_token_seconds = time.perf_counter() - started
                    parts.append(content)
    answer = "".join(parts).strip()
    if not answer or first_token_seconds is None:
        raise RuntimeError("The server returned no answer text.")
    if not usage or "prompt_tokens" not in usage:
        raise RuntimeError("The server did not return prompt token usage.")
    return {
        "answer": answer,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage.get("completion_tokens"),
        "ttft_seconds": round(first_token_seconds, 4),
    }


def save(state, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("turn1", "turn2", "fresh", "self-test"))
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--metrics", default="http://127.0.0.1:8080/metrics")
    parser.add_argument("--state", type=Path, default=STATE)
    args = parser.parse_args()

    if args.step == "self-test":
        doc = document()
        assert 1800 < len(doc.split()) < 3000
        assert turn_two_messages(doc, "Mira Solis")[:2] == turn_one_messages(doc)
        print(f"Self-test passed: {len(doc.split())} document words")
        return

    if args.step == "turn1":
        doc = document()
        result = chat(args.endpoint, turn_one_messages(doc))
        state = {"model": MODEL, "document": doc, "turn1": result}
        save(state, args.state)
        print(json.dumps({"turn1": result, "state": str(args.state)}, indent=2))
        return

    state = json.loads(args.state.read_text(encoding="utf-8"))
    if state["model"] != MODEL:
        raise RuntimeError("Saved state uses a different model.")
    messages = turn_two_messages(state["document"], state["turn1"]["answer"])
    if args.step == "turn2":
        before = metric_total(args.metrics)
        result = chat(args.endpoint, messages)
        after = metric_total(args.metrics)
        for _ in range(25):
            if after > before:
                break
            time.sleep(0.2)
            after = metric_total(args.metrics)
        result["lmcache_hit_tokens"] = round(after - before)
        state["turn2"] = result
    else:
        result = chat(args.endpoint, messages)
        state["fresh_prefill"] = result
    save(state, args.state)
    print(json.dumps({args.step: result, "state": str(args.state)}, indent=2))
    if args.step == "turn2" and result["lmcache_hit_tokens"] <= 0:
        raise SystemExit("No LMCache hit was observed; inspect the connector and server logs.")


if __name__ == "__main__":
    main()
