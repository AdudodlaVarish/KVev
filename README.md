# KVev: two-turn KV cache smoke test

This first experiment checks whether a second question can retrieve the shared document prefix from LMCache. It uses Meta Llama 3.2 1B Instruct on the local 8 GB GPU. It does not use lossy compression or a Jev policy yet.

## Environment

Run in the WSL distribution named Ubuntu, from /home/varish/jevkv. The project virtual environment has Python 3.13, vLLM 0.30.0, and LMCache 0.5.5. The model is gated: accept access on [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) and sign in locally with .venv/bin/hf auth login. Keep tokens out of this repository.

The test writes its results to results/two_turn.json, which is ignored by Git. Run the local check first:

~~~bash
cd ~/jevkv
source .venv/bin/activate
bash scripts/check_env.sh
python scripts/two_turn.py self-test
~~~

## Run the cache test

Use three Ubuntu terminals. Keep Terminal A running throughout the cache test. LMCache provides CPU memory for the retained KV prefix. On this WSL installation, the default CUDA IPC transfer failed at KV registration. The engine-driven mode below avoids CUDA IPC and uses a slower copy path.

**Terminal A — LMCache**

~~~bash
cd ~/jevkv
source .venv/bin/activate
lmcache server --host localhost --port 5555 \
  --l1-size-gb 2 --eviction-policy LRU --chunk-size 256 \
  --supported-transfer-mode engine_driven
~~~

**Terminal B — vLLM with LMCache**

~~~bash
cd ~/jevkv
source .venv/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.75 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"localhost","lmcache.mp.port":5555,"lmcache.mp.mp_transfer_mode":"engine_driven"}}'
~~~

The sampler setting avoids a FlashInfer warmup step that needs a local `ninja` executable. Wait until vLLM is ready. Its first start downloads the gated model into the Hugging Face cache if it is not already present.

**Terminal C — first question**

~~~bash
cd ~/jevkv
source .venv/bin/activate
python scripts/two_turn.py turn1
~~~

Stop only vLLM in Terminal B with Ctrl+C, then start it again with the same command. Keep LMCache in Terminal A running. This removes vLLM's GPU prefix cache while retaining LMCache's CPU copy.

**Terminal C — second question**

~~~bash
python scripts/two_turn.py turn2
~~~

The script saves both answers, prompt token counts, times to first token, and the increase in LMCache's `lmcache_mp_lookup_hit_tokens_total` counter at `http://127.0.0.1:8080/metrics`. A positive `lmcache_hit_tokens` value shows the second request found a prefix outside vLLM's restarted GPU cache.

## Fresh-prefill comparison

Stop vLLM again and restart it without the LMCache connector:

~~~bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --host 127.0.0.1 --port 8000 \
  --max-model-len 4096 --gpu-memory-utilization 0.75
~~~

Then run:

~~~bash
python scripts/two_turn.py fresh
~~~

This sends the same second-turn prompt with no external KV reuse and records its first-token time. Server startup is outside the timed requests. One run is a smoke test, not a performance claim.

## First local result

On 2026-09-22, the 2,878-token document produced a 2,937-token first-turn prompt. Turn 1 answered that Mira Solis founded Meridian Labs, and LMCache stored 2,816 tokens. After restarting vLLM with LMCache still running, Turn 2 used a 2,973-token prompt, retrieved **2,816 tokens**, and completed its comparison. Its time to first token was 0.862 seconds. The fresh-prefill reference for the same prompt was 0.528 seconds. These single-run timings do not establish a speed advantage for the engine-driven copy path.

## Milestone 2: FP8 compressed L2 reuse

`scripts/compressed_reuse.py` runs four distinct documents through fresh prefill, raw L2, and FP8 L2. Each document is 2.8–3.0K tokens and asks for a specific phrase, number, comparison, or exception. The script saves the original document and token IDs, full answers, prompt lengths, first-token times, L1/L2 hit counts, and L2 data-file bytes in `results/compressed_reuse.json`.

Run `python scripts/compressed_reuse.py init` once. Start vLLM **without** the LMCache connector using the fresh-prefill command above, then run `python scripts/compressed_reuse.py fresh` and stop vLLM.

For the raw condition, start LMCache in Terminal A:

~~~bash
mkdir -p ~/jevkv/results/l2/raw
lmcache server --host localhost --port 5555 \
  --l1-size-gb 2 --eviction-policy noop --chunk-size 256 \
  --supported-transfer-mode engine_driven --l2-store-policy skip_l1 \
  --l2-adapter '{"type":"fs","base_path":"/home/varish/jevkv/results/l2/raw"}'
~~~

Start vLLM in Terminal B with the LMCache connector command above. Run `python scripts/compressed_reuse.py raw-seed`, restart **only vLLM**, then run `python scripts/compressed_reuse.py raw-query`. Stop both servers.

For FP8, use a separate empty directory and the same vLLM command:

~~~bash
mkdir -p ~/jevkv/results/l2/fp8
lmcache server --host localhost --port 5555 \
  --l1-size-gb 2 --eviction-policy noop --chunk-size 256 \
  --supported-transfer-mode engine_driven --l2-store-policy skip_l1 \
  --l2-adapter '{"type":"fs","base_path":"/home/varish/jevkv/results/l2/fp8","serde":{"type":"fp8","fp8_dtype":"float8_e4m3fn"}}'
~~~

Run `python scripts/compressed_reuse.py fp8-seed`, restart only vLLM, then run `python scripts/compressed_reuse.py fp8-query` and `python scripts/compressed_reuse.py report`. The query steps fail if L2 retrieves no tokens or L1 retrieves any. Stop both servers when finished. To repeat the full experiment, clear only the generated `results/l2/raw` and `results/l2/fp8` directories, then run `init` again.

In the first local run, **all four questions passed their fixed fact checks** in all three conditions. Raw L2 used 369,098,752 data bytes (352 MiB); FP8 used 184,549,376 bytes (176 MiB), a 50% reduction. Each cached second turn retrieved 2,816 tokens from L2 and zero from L1 after a vLLM restart. FP8 first-token times ranged from 0.466 to 0.768 seconds; fresh times ranged from 0.315 to 0.688 seconds. These are smoke-test observations, not a speed claim. LMCache's [L2 serde](https://docs.lmcache.ai/mp/serde.html) performs FP8 conversion on disk writes and reads; active GPU KV is reconstructed.
