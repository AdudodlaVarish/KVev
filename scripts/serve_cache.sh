#!/usr/bin/env bash
set -euo pipefail
cd /home/varish/jevkv
L2_DIR="$(realpath -m "${JEVKV_L2_PATH:-results/serve_l2/fp8}")"
TRANSFER_MODE="${JEVKV_TRANSFER_MODE:-engine_driven}"
SHM_ARGS=()
if [[ "$TRANSFER_MODE" == engine_driven ]]; then
  SHM_ARGS=(--shm-name jevkv_serve_pool)
fi
mkdir -p "$L2_DIR"
exec .venv/bin/lmcache server --host localhost --port 5555 \
  --l1-size-gb "${JEVKV_L1_GB:-3}" --eviction-policy LRU --chunk-size 256 \
  --supported-transfer-mode "$TRANSFER_MODE" "${SHM_ARGS[@]}" \
  --l2-store-policy default \
  --l2-prefetch-policy retain \
  --l2-adapter "{\"type\":\"fs\",\"base_path\":\"$L2_DIR\",\"serde\":{\"type\":\"fp8\",\"fp8_dtype\":\"float8_e4m3fn\"}}"
