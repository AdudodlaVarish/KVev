#!/usr/bin/env bash
set -euo pipefail
cd /home/varish/jevkv
mkdir -p results/laya_l2/fp8 results/laya_l2/tq4
exec .venv/bin/python scripts/laya_cache_server.py server --host localhost --port 5555 \
  --l1-size-gb 2 --eviction-policy noop --chunk-size 256 \
  --supported-transfer-mode engine_driven --l2-store-policy jevkv_codec \
  --l2-prefetch-policy jevkv_codec \
  --l2-adapter '{"type":"fs","base_path":"/home/varish/jevkv/results/laya_l2/fp8","serde":{"type":"fp8","fp8_dtype":"float8_e4m3fn"}}' \
  --l2-adapter '{"type":"fs","base_path":"/home/varish/jevkv/results/laya_l2/tq4","serde":{"type":"turboquant_packed","preset":"turboquant_4bit_nc","head_dim":64,"block_size":16,"skip_first_layers":0,"skip_last_layers":0}}'
