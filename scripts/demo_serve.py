"""Send a long document and a follow-up to the local JevKV chat endpoint."""

import json
import secrets
from urllib.request import Request, urlopen

from two_turn import MODEL

ENDPOINT = "http://127.0.0.1:8001/v1/chat/completions"


def document():
    sections = []
    background = (
        "The research team recorded equipment checks, meeting notes, review dates, "
        "and routine project updates. These details provide context for the archive. "
    )
    for number in range(1, 241):
        if number == 2:
            fact = "Mira Solis founded Meridian Labs in 2017. "
        elif number == 8:
            fact = "Mira Solis promised to keep customer data private. "
        else:
            fact = "The team updated its internal research archive. "
        sections.append(f"Section {number}: {fact}{background * 4}")
    return f"Document ID: {secrets.token_hex(8)}\n" + "\n".join(sections)


def ask(messages):
    request = Request(
        ENDPOINT,
        data=json.dumps({"model": MODEL, "messages": messages,
                         "temperature": 0, "max_tokens": 80}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request, timeout=180) as response:
        result = json.load(response)
        route = response.headers.get("X-JevKV-Route")
    return result["choices"][0]["message"]["content"].strip(), route, result["usage"]


def main():
    messages = [
        {"role": "system", "content": "Answer from the document in one sentence."},
        {"role": "user", "content":
         f"Document:\n{document()}\n\nQuestion: Who founded Meridian Labs?"},
    ]
    answer, route, usage = ask(messages)
    print(f"Turn 1 | route={route} | prompt={usage['prompt_tokens']} tokens\n{answer}\n")
    if route != "seed" or "Mira Solis" not in answer:
        raise SystemExit("The first turn did not seed and answer the document question.")
    messages += [
        {"role": "assistant", "content": answer},
        {"role": "user", "content": "What did Mira Solis promise about customer data in section 8?"},
    ]
    answer, route, usage = ask(messages)
    print(f"Turn 2 | route={route} | prompt={usage['prompt_tokens']} tokens\n{answer}")
    if route != "reuse" or "private" not in answer.lower():
        raise SystemExit("The second turn did not reuse and answer the document question.")


if __name__ == "__main__":
    main()
