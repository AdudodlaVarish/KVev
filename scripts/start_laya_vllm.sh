#!/usr/bin/env bash
set -euo pipefail
cd /home/varish/jevkv
export PYTHONPATH=scripts
export VLLM_USE_FLASHINFER_SAMPLER=0
exec .venv/bin/vllm serve meta-llama/Llama-3.2-1B-Instruct \
  --host 127.0.0.1 --port 8000 --max-model-len 9216 \
  --gpu-memory-utilization 0.75 \
  --kv-transfer-config '{"kv_connector":"JevLMCacheMPConnector","kv_connector_module_path":"jev_connector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"localhost","lmcache.mp.port":5555,"lmcache.mp.mp_transfer_mode":"engine_driven"}}'
