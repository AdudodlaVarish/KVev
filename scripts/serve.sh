#!/usr/bin/env bash
set -euo pipefail
cd /home/varish/jevkv

LOG_DIR=results/serve_runtime
mkdir -p "$LOG_DIR"
for port in 8000 8001 5555 8080; do
  if (echo >"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    echo "Port $port is already in use. Stop the existing JevKV service first." >&2
    exit 1
  fi
done

cache_pid=
vllm_pid=
proxy_pid=
cleanup() {
  trap - EXIT INT TERM
  for pid in "$proxy_pid" "$vllm_pid" "$cache_pid"; do
    if [[ -n "$pid" ]]; then kill "$pid" 2>/dev/null || true; fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

bash scripts/serve_cache.sh >"$LOG_DIR/cache.log" 2>&1 & cache_pid=$!
echo "Starting cache..."
for _ in {1..120}; do
  if curl --silent --fail http://127.0.0.1:8080/healthcheck >/dev/null; then break; fi
  if ! kill -0 "$cache_pid" 2>/dev/null; then
    echo "Cache failed to start; see $LOG_DIR/cache.log" >&2
    exit 1
  fi
  sleep 1
done
if ! curl --silent --fail http://127.0.0.1:8080/healthcheck >/dev/null; then
  echo "Cache startup timed out; see $LOG_DIR/cache.log" >&2
  exit 1
fi

bash scripts/serve_vllm.sh >"$LOG_DIR/vllm.log" 2>&1 & vllm_pid=$!
echo "Loading Llama and starting vLLM..."
for _ in {1..300}; do
  if curl --silent --fail http://127.0.0.1:8000/health >/dev/null; then break; fi
  if ! kill -0 "$vllm_pid" 2>/dev/null; then
    echo "vLLM failed to start; see $LOG_DIR/vllm.log" >&2
    exit 1
  fi
  sleep 1
done
if ! curl --silent --fail http://127.0.0.1:8000/health >/dev/null; then
  echo "vLLM startup timed out; see $LOG_DIR/vllm.log" >&2
  exit 1
fi

.venv/bin/python scripts/serve_proxy.py >"$LOG_DIR/proxy.log" 2>&1 & proxy_pid=$!
for _ in {1..30}; do
  if curl --silent --fail http://127.0.0.1:8001/health >/dev/null; then break; fi
  if ! kill -0 "$proxy_pid" 2>/dev/null; then
    echo "Proxy failed to start; see $LOG_DIR/proxy.log" >&2
    exit 1
  fi
  sleep 1
done
if ! curl --silent --fail http://127.0.0.1:8001/health >/dev/null; then
  echo "Proxy startup timed out; see $LOG_DIR/proxy.log" >&2
  exit 1
fi

echo "JevKV is ready at http://127.0.0.1:8001/v1"
echo "Run: .venv/bin/python scripts/demo_serve.py"
echo "Logs: $LOG_DIR (Ctrl+C stops all services)"
wait -n "$cache_pid" "$vllm_pid" "$proxy_pid"
echo "A JevKV service stopped; see $LOG_DIR/*.log" >&2
exit 1
