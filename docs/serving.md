# Local serving

KVev serves Meta Llama 3.2 1B Instruct through a local, OpenAI-compatible chat endpoint. It uses vLLM for inference and LMCache for reusable KV. The live routing policy uses simple question rules; LAYA is used only in offline experiments.

## Start

In Ubuntu, with the project `.venv` and model access already set up:

```bash
cd ~/jevkv
bash scripts/serve.sh
```

Wait for the ready message. The launcher starts LMCache, vLLM, and the proxy. Press Ctrl+C to stop them. Logs are in `results/serve_runtime/`.

## Try it

In another Ubuntu terminal:

```bash
cd ~/jevkv
.venv/bin/python scripts/demo_serve.py
```

The demo sends a long document, then asks a follow-up using the same message prefix.

## Chat endpoint

- Base URL: `http://127.0.0.1:8001/v1`
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- `POST /v1/chat/completions` accepts text chat with streaming or non-streaming responses.
- `GET /health` reports whether vLLM and LMCache are ready.

Long first turns seed the cache. Simple follow-ups can reuse it; sensitive, unsupported, or short requests are recomputed. The `X-JevKV-Route` response header shows `seed`, `reuse`, or `fresh`. Send `X-JevKV-Cache: bypass` to force a fresh request without storing its KV. Keep the earlier messages byte-for-byte identical to reuse their prefix.

The service binds to localhost. Its default maximum context is 32,768 tokens; set `JEVKV_MAX_LEN` before starting to change it. Saved benchmark details are in [milestone6.json](results/milestone6.json).
