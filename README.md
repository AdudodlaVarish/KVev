# KVev

KVev is a local, cache-aware chat server for long documents. Its OpenAI-compatible endpoint uses simple rules to decide whether a follow-up reuses a stored KV prefix or recomputes it from the original text. LAYA is used in offline routing experiments, not in the live server.

## Built with

- **vLLM** serves Meta Llama 3.2 1B Instruct.
- **LMCache** stores KV in CPU memory and FP8-compressed local storage.
- **FastAPI** provides the local chat endpoint.
- **LAYA** evaluates question sensitivity in optional offline experiments.

## Run locally

From Ubuntu, with the project environment set up:

```bash
cd ~/jevkv
bash scripts/serve.sh
```

In another terminal, run `.venv/bin/python scripts/demo_serve.py` for a two-turn example. The endpoint is `http://127.0.0.1:8001/v1/chat/completions`. See [the serving guide](docs/serving.md) for setup and usage details.

## TTFT results and outlook

On an 8 GB RTX 4060 laptop with Llama 1B, an exploratory three-question, 8K-token test using **uncompressed** LMCache L2 reached **0.916 s median time to first token**, versus **1.543 s** for fresh prefill; all three fixed fact checks passed. 

**Larger-model estimate:** Reusing long prefixes after they have left GPU cache should have more value when prefill is expensive, but larger KV transfers could offset it. As a reference point, [LMCache's separate Qwen3-8B long-document benchmark](https://docs.lmcache.ai/getting_started/benchmarking.html) reported **757 ms → 185 ms mean TTFT** with CPU offloading.
