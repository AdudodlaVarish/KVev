# JevKV: two-turn KV cache experiments

This project measures whether a second question can retrieve a shared document prefix from LMCache. It uses Meta Llama 3.2 1B Instruct on the local 8 GB GPU. The first two milestones establish L2 retrieval and FP8 compression, the third benchmarks TurboQuant, and the fourth adds a rule-based request-time reuse decision.

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

## Milestone 3: length, question, and codec benchmark

`scripts/benchmark.py` generates 27 distinct documents: three examples each of easy lookup, exact phrase, and two-section comparison at approximately 2K, 4K, and 8K second-turn prompt tokens. It saves the original text and token IDs in ignored `results/benchmark.json`. All four conditions use the same second-turn prompts and deterministic decoding: fresh prefill, raw L2, FP8 L2, and TurboQuant 4-bit L2. Each cached condition has its own empty L2 directory. Seed it, restart **only vLLM**, then query it while LMCache stays running. The script rejects a cached query unless L2 retrieves tokens and L1 retrieves zero tokens.

Activate `.venv` and run `python scripts/benchmark.py init` once. For fresh prefill, start vLLM without the connector as below, then run `python scripts/benchmark.py fresh` and stop vLLM:

~~~bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --host 127.0.0.1 --port 8000 --max-model-len 9216 \
  --gpu-memory-utilization 0.75 --no-enable-prefix-caching
~~~

For each L2 condition, start a separate LMCache server and use this same vLLM command in another terminal:

~~~bash
VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --host 127.0.0.1 --port 8000 --max-model-len 9216 \
  --gpu-memory-utilization 0.75 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"localhost","lmcache.mp.port":5555,"lmcache.mp.mp_transfer_mode":"engine_driven"}}'
~~~

Start LMCache for **raw** L2:

~~~bash
mkdir -p results/benchmark_l2/raw
lmcache server --host localhost --port 5555 --l1-size-gb 2 \
  --eviction-policy noop --chunk-size 256 \
  --supported-transfer-mode engine_driven --l2-store-policy skip_l1 \
  --l2-adapter '{"type":"fs","base_path":"/home/varish/jevkv/results/benchmark_l2/raw"}'
~~~

Run `python scripts/benchmark.py seed --mode raw`, restart only vLLM, then run `python scripts/benchmark.py query --mode raw`. Stop both servers. Repeat with **FP8**, using:

~~~bash
mkdir -p results/benchmark_l2/fp8
lmcache server --host localhost --port 5555 --l1-size-gb 2 \
  --eviction-policy noop --chunk-size 256 \
  --supported-transfer-mode engine_driven --l2-store-policy skip_l1 \
  --l2-adapter '{"type":"fs","base_path":"/home/varish/jevkv/results/benchmark_l2/fp8","serde":{"type":"fp8","fp8_dtype":"float8_e4m3fn"}}'
~~~

Run the matching `seed --mode fp8` and `query --mode fp8` commands with a vLLM restart between them. Stop both servers. For **TurboQuant**, this vLLM/LMCache combination stores packed 3D KV chunks, while LMCache's built-in TurboQuant serializer expects 4D chunks. `scripts/turboquant_packed.py` adapts the shape and delegates compression to LMCache's serializer. Run its `self-test` before starting its server:

~~~bash
python scripts/turboquant_packed.py self-test
mkdir -p results/benchmark_l2/turbo
python scripts/turboquant_packed.py server --host localhost --port 5555 \
  --l1-size-gb 2 --eviction-policy noop --chunk-size 256 \
  --supported-transfer-mode engine_driven --l2-store-policy skip_l1 \
  --l2-adapter '{"type":"fs","base_path":"/home/varish/jevkv/results/benchmark_l2/turbo","serde":{"type":"turboquant_packed","preset":"turboquant_4bit_nc","head_dim":64,"block_size":16,"skip_first_layers":0,"skip_last_layers":0}}'
~~~

Run the matching `seed --mode turbo` and `query --mode turbo` commands with a vLLM restart between them. Finally run `python scripts/benchmark.py report`. The script waits for each seed's L2 write to finish before adding the next document; this prevents the 2 GB L1 staging buffer from overflowing during TurboQuant compression. To rerun from scratch, remove only generated `results/benchmark.json`, `results/benchmark_summary.json`, and `results/benchmark_l2`, then initialize again.

### Local result (2026-09-22)

All 81 cached follow-ups retrieved from L2 after vLLM restarts, with zero L1 hit tokens. Hits were 1,792, 3,840, and 7,936 tokens for the 2K, 4K, and 8K cases respectively. The table counts answers containing the fixed expected facts; full answers, per-case timings, and the nine length/question cells are in the ignored result files.

| Condition | Fact checks | L2 data bytes | Median time to first token |
| --- | ---: | ---: | ---: |
| Fresh prefill | 27/27 | — | 0.4459 s |
| Raw L2 | 27/27 | 4,001,366,016 | 0.5147 s |
| FP8 L2 | 27/27 | 2,000,683,008 (50% of raw) | 0.5186 s |
| TurboQuant 4-bit L2 | 26/27 | 1,094,123,520 (27% of raw) | 1.2812 s |

TurboQuant's failed case was an 8K comparison: its answer mentioned the section 2 audit-results commitment but missed the section 8 commitment to withhold employee identities. FP8 answers differed in wording from raw for 4 of 27 cases; TurboQuant differed for 11. The fact checks use required phrases, so they are useful for these known facts but do not grade every nuance of an answer. These timings are observations on one WSL laptop with engine-driven transfer and three examples per length/question cell. They show no overall first-token speed benefit from L2 reuse in this setup; TurboQuant's codec cost was substantial.

## Milestone 4: first Jev request-time router

`scripts/jev_router.py` uses the same 27 saved documents and token IDs from `results/benchmark.json`. Its rule reads the question text: a simple lookup naming one section uses FP8 L2; exact-wording questions, comparisons, multiple-section questions, and unknown forms get fresh prefill. The rule does not read benchmark question labels. This is a conservative integration baseline, not a trained quality predictor.

The fresh route sends the **same prompt** with a new `cache_salt`. vLLM and LMCache include that salt in cache keys, so the seeded unsalted prefix cannot hit. The small `scripts/jev_connector.py` extension also removes store operations for salted requests. Their KV is computed for the current request and is not written to LMCache. The script first checks this with a salted request against a seeded prompt, then refuses any routed run where FP8 fails to retrieve solely from L2 or fresh retrieves, submits a store, or increases L2 data bytes.

Run `python scripts/jev_router.py self-test` and keep `results/benchmark.json` from Milestone 3. Start a **new LMCache server** with an empty directory and the same FP8 settings:

~~~bash
mkdir -p results/router_l2/fp8
lmcache server --host localhost --port 5555 --l1-size-gb 2 \
  --eviction-policy noop --chunk-size 256 \
  --supported-transfer-mode engine_driven --l2-store-policy skip_l1 \
  --l2-adapter '{"type":"fs","base_path":"/home/varish/jevkv/results/router_l2/fp8","serde":{"type":"fp8","fp8_dtype":"float8_e4m3fn"}}'
~~~

Start vLLM with the project connector, both while seeding and after the restart:

~~~bash
PYTHONPATH=/home/varish/jevkv/scripts VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --host 127.0.0.1 --port 8000 --max-model-len 9216 \
  --gpu-memory-utilization 0.75 \
  --kv-transfer-config '{"kv_connector":"JevLMCacheMPConnector","kv_connector_module_path":"jev_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"localhost","lmcache.mp.port":5555,"lmcache.mp.mp_transfer_mode":"engine_driven"}}'
~~~

Run `python scripts/jev_router.py seed`. Restart **only vLLM** with the same command, leaving LMCache running. Run `python scripts/jev_router.py run` followed by `python scripts/jev_router.py report`. The ignored `results/jev_router.json` saves each decision, reason, answer, prompt length, fact check, retrieval counters, store counter, and first-token time; `results/jev_router_summary.json` contains the comparison. To repeat the experiment, stop both servers and clear only the generated router state files and `results/router_l2/fp8` directory before starting again.

### Local result (2026-09-23)

The salt preflight retrieved zero L2 tokens, submitted zero stores, and added zero L2 data bytes. The router chose FP8 for 9 questions and fresh prefill for 18. Every FP8 request retrieved 1,792, 3,840, or 7,936 tokens from L2. Every fresh request retrieved zero, submitted zero stores, and added zero L2 data bytes. L1 hits were zero throughout, and all **27/27** answers passed the fixed fact checks. The routed median time to first token was **0.4484 s**; the earlier always-fresh and always-FP8 medians were **0.4459 s** and **0.5186 s**. These are exploratory comparisons because server configurations and run order differed. This run proves routing, not a latency or accuracy gain.

The FP8 seed snapshot used **2,000,683,008 bytes** of L2 data. After the restart and query run, the directory held **2,038,431,744 bytes**. Unsalted FP8 follow-up requests submitted seven stores; two further chunks were written outside those measured request intervals. No salted fresh request submitted a store or added L2 data. The original compressed prefixes remain available for later reuse.
