#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. /etc/os-release
printf 'Distribution: %s\n' "${PRETTY_NAME:-unknown}"

if grep -qi microsoft /proc/sys/kernel/osrelease; then
  printf 'WSL: available\n'
else
  printf 'WSL: not detected\n' >&2
  exit 1
fi

printf 'System Python: '
python3 --version

if [[ -x "$project_dir/.venv/bin/python" ]]; then
  printf 'Project Python: '
  "$project_dir/.venv/bin/python" --version
  "$project_dir/.venv/bin/python" -c 'import importlib.metadata as m; print("vLLM:", m.version("vllm")); print("LMCache:", m.version("lmcache")); print("PyTorch:", m.version("torch"))'
else
  printf 'Project Python: .venv not found\n' >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  printf 'uv: '
  uv --version
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  printf 'uv: '
  "$HOME/.local/bin/uv" --version
else
  printf 'uv: not found\n' >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  printf 'GPU: '
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
  printf 'GPU: nvidia-smi not found\n' >&2
  exit 1
fi

df -h "$HOME" | awk 'NR == 2 { printf "Home filesystem: %s total, %s available\n", $2, $4 }'
