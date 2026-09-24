"""Local chat proxy that admits long prefixes to LMCache and skips sensitive reuse."""

import argparse
import asyncio
import json
import re
import secrets
import time
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from cache_policies import SALTS
from two_turn import MODEL

LOOKUP = re.compile(r"^(?:what|which|who|where|when|how many)\b", re.I)
SENSITIVE = re.compile(
    r"\b(?:exact|verbatim|quote|wording|precise|compare|contrast|difference|"
    r"differences|both|all|every|list|enumerate|explain|summarize|summary|why)\b", re.I
)
SECTIONS = re.compile(r"\bsections?\s+(\d+(?:\s*(?:,|and|&)\s*\d+)*)", re.I)
FRESH = "fresh"
REUSE = "reuse"
SEED = "seed"
CALIBRATION = Path(__file__).resolve().parents[1] / "results" / "serve_calibration.json"


def token_count(tokenizer, messages, generation_prompt):
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation_prompt
    )
    return len(ids["input_ids"] if isinstance(ids, Mapping) else ids)


def simple_lookup(content):
    question = content.splitlines()[0].strip()
    question = re.sub(r"^question:\s*", "", question, flags=re.I)
    if SENSITIVE.search(question):
        return False, "sensitive question"
    sections = [number for group in SECTIONS.findall(question)
                for number in re.findall(r"\d+", group)]
    if len(set(sections)) > 1:
        return False, "multiple sections"
    if LOOKUP.match(question):
        return True, "simple lookup"
    return False, "unrecognized question"


def route(payload, tokenizer, min_prefix_tokens):
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return FRESH, "invalid messages", 0, 0
    if payload.get("model") != MODEL:
        return FRESH, "different model", 0, 0
    if any(not isinstance(item, dict) or not isinstance(item.get("content"), str)
           or item.get("role") not in ("system", "user", "assistant") for item in messages):
        return FRESH, "non-text chat", 0, 0
    try:
        full_tokens = token_count(tokenizer, messages, True)
    except (TypeError, ValueError, KeyError):
        return FRESH, "unsupported prompt", 0, 0
    if messages[-1]["role"] != "user" or payload.get("tools") or payload.get("tool_choice"):
        return FRESH, "unsupported chat shape", full_tokens, 0
    if payload.get("response_format") or payload.get("logprobs"):
        return FRESH, "format-sensitive response", full_tokens, 0
    prior_users = sum(item["role"] == "user" for item in messages[:-1])
    if not prior_users:
        if full_tokens >= min_prefix_tokens:
            return SEED, "long first turn", full_tokens, 0
        return FRESH, "short first turn", full_tokens, 0
    prefix_tokens = token_count(tokenizer, messages[:-1], False)
    if prefix_tokens < min_prefix_tokens:
        return FRESH, "short shared prefix", full_tokens, prefix_tokens
    eligible, reason = simple_lookup(messages[-1]["content"])
    return (REUSE if eligible else FRESH), reason, full_tokens, prefix_tokens


def log_request(route_name, reason, prompt_tokens, prefix_tokens, started, ttft=None,
                status=200):
    print(json.dumps({
        "route": route_name,
        "reason": reason,
        "prompt_tokens": prompt_tokens,
        "prefix_tokens": prefix_tokens,
        "ttft_seconds": round(ttft, 4) if ttft is not None else None,
        "total_seconds": round(time.perf_counter() - started, 4),
        "status": status,
    }), flush=True)


def create_app(backend="http://127.0.0.1:8000", cache="http://127.0.0.1:8080",
               min_prefix_tokens=4096, tokenizer=None, client=None):
    @asynccontextmanager
    async def lifespan(app):
        nonlocal tokenizer, client
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        owns_client = client is None
        if owns_client:
            client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=10))
        yield
        if owns_client:
            await client.aclose()

    app = FastAPI(title="JevKV", lifespan=lifespan)

    @app.get("/health")
    async def health():
        try:
            upstream, lmcache = await asyncio.gather(
                client.get(backend + "/health", timeout=3),
                client.get(cache + "/healthcheck", timeout=3),
            )
            if upstream.is_success and lmcache.is_success:
                return {"status": "ready"}
        except httpx.HTTPError:
            pass
        return JSONResponse({"status": "unavailable"}, status_code=503)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        started = time.perf_counter()
        try:
            payload = await request.json()
        except ValueError:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)
        if not isinstance(payload, dict) or payload.get("model") != MODEL:
            return JSONResponse({"error": f"Only model {MODEL} is supported"}, status_code=400)
        if "cache_salt" in payload:
            return JSONResponse({"error": "cache_salt is managed by JevKV"}, status_code=400)
        hint = request.headers.get("x-jevkv-cache")
        if hint not in (None, "bypass"):
            return JSONResponse({"error": "X-JevKV-Cache must be bypass"}, status_code=400)
        route_name, reason, prompt_tokens, prefix_tokens = route(
            payload, tokenizer, min_prefix_tokens)
        if hint == "bypass":
            route_name, reason = FRESH, "client bypass"
        payload["cache_salt"] = (SALTS["fp8"] if route_name in (SEED, REUSE)
                                 else secrets.token_urlsafe(32))
        stream = payload.get("stream") is True
        try:
            upstream = await client.send(client.build_request(
                "POST", backend + "/v1/chat/completions", json=payload,
                timeout=httpx.Timeout(180, connect=10),
            ), stream=stream)
        except httpx.HTTPError as exc:
            log_request(route_name, reason, prompt_tokens, prefix_tokens, started,
                        status=502)
            return JSONResponse({"error": f"vLLM unavailable: {type(exc).__name__}"},
                                status_code=502)
        if not upstream.is_success:
            body = await upstream.aread()
            await upstream.aclose()
            log_request(route_name, reason, prompt_tokens, prefix_tokens, started,
                        status=upstream.status_code)
            return Response(content=body, status_code=upstream.status_code,
                            media_type=upstream.headers.get("content-type"))
        headers = {"X-JevKV-Route": route_name}
        if not stream:
            body = await upstream.aread()
            await upstream.aclose()
            log_request(route_name, reason, prompt_tokens, prefix_tokens, started,
                        ttft=time.perf_counter() - started)
            return Response(content=body, status_code=200,
                            media_type=upstream.headers.get("content-type"), headers=headers)

        async def events():
            ttft = None
            status = 200
            try:
                async for line in upstream.aiter_lines():
                    if ttft is None and line.startswith("data: "):
                        try:
                            event = json.loads(line[6:])
                            if any(choice.get("delta", {}).get("content")
                                   for choice in event.get("choices", [])):
                                ttft = time.perf_counter() - started
                        except (ValueError, TypeError):
                            pass
                    yield (line + "\n").encode("utf-8")
            except httpx.HTTPError:
                status = 502
                raise
            finally:
                await upstream.aclose()
                log_request(route_name, reason, prompt_tokens, prefix_tokens,
                            started, ttft, status)

        return StreamingResponse(events(), media_type="text/event-stream", headers=headers)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--cache", default="http://127.0.0.1:8080")
    parser.add_argument("--min-prefix-tokens", type=int)
    args = parser.parse_args()
    threshold = args.min_prefix_tokens
    if threshold is None:
        threshold = (json.loads(CALIBRATION.read_text())["min_prefix_tokens"]
                     if CALIBRATION.exists() else 4096)
    print(json.dumps({"min_prefix_tokens": threshold,
                      "calibrated": CALIBRATION.exists() and args.min_prefix_tokens is None}),
          flush=True)
    import uvicorn
    uvicorn.run(create_app(args.backend, args.cache, threshold),
                host=args.host, port=args.port)


if __name__ == "__main__":
    main()
