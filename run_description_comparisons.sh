#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ ${1:-} == --preflight-only && $# -eq 1 ]]; then
  preflight_only=1
elif [[ $# -eq 0 ]]; then
  preflight_only=0
else
  printf 'Usage: %s [--preflight-only]\n' "$0" >&2
  exit 2
fi

python=.venv/bin/python
expected_hash=4ad4c1a4b4ea2f7adc48c236715ff43f13938a98e16a58f0b584a1e0facb69e8
models=(
  configs/models/gpt6_luna_decisions.yaml
  configs/models/microsoft_decision1_openrouter.yaml
  configs/models/d3_gguf_q4.yaml
)

if [[ ! -x $python ]]; then
  printf 'Missing %s; set up the project environment first.\n' "$python" >&2
  exit 1
fi

actual_hash=$("$python" - <<'PY'
import hashlib
from pathlib import Path

print(hashlib.sha256(Path('data/validation_sharks.csv').read_bytes()).hexdigest())
PY
)
if [[ $actual_hash != "$expected_hash" ]]; then
  printf 'Shark CSV hash differs from Run 2; stopping before any model calls.\n' >&2
  exit 1
fi

if ! curl -fsS --max-time 5 http://127.0.0.1:8080/health >/dev/null; then
  printf 'd3 server is unavailable at http://127.0.0.1:8080.\n' >&2
  exit 1
fi
if ! props=$(curl -fsS --max-time 5 http://127.0.0.1:8080/props); then
  printf 'Could not read d3 server properties.\n' >&2
  exit 1
fi
if ! context=$(printf '%s' "$props" | "$python" -c '
import json
import sys

props = json.load(sys.stdin)
model = props.get("model_path", "")
slots = props.get("total_slots")
context = props.get("default_generation_settings", {}).get("n_ctx")
if not isinstance(model, str) or not model.endswith("vllm-d3-Q4_K_M.gguf"):
    raise SystemExit("Wrong d3 model loaded")
if slots != 1 or type(context) is not int:
    raise SystemExit("Expected one d3 slot with a reported context size")
print(context)
'); then
  printf 'd3 server properties do not match the configured model and one-slot setup.\n' >&2
  exit 1
fi
if (( context < 8192 )); then
  cat >&2 <<'EOF'
d3 has fewer than 8192 context tokens. Restart the server with the same model and:
  -c 8192 -b 8192 -ub 8192 -np 1 --no-cont-batching
See docs/D3_LOCAL_MAC.md for the full llama-server command, then rerun this script.
EOF
  exit 1
fi

run_condition() {
  local shot=$1
  local mode=$2
  local args=(--task sharks --include-description)
  if [[ $shot == zero-shot ]]; then
    args+=(--zero-shot)
  fi
  local model_args=()
  if [[ $mode == d3-only ]]; then
    model_args=(--model "${models[2]}" --score-only)
  else
    model_args=(--model "${models[0]}" --model "${models[1]}" --model "${models[2]}")
  fi
  if [[ $mode == dry-run ]]; then
    model_args+=(--dry-run)
  fi
  printf '\n%s: %s\n' "$shot" "$mode"
  PYTHONPATH=src "$python" -m wildlife_screening.cli run \
    "${args[@]}" "${model_args[@]}" --cache outputs/cache \
    --concurrency 1 --spend-ceiling 1
}

printf 'Preflight: fixed shark data, d3 server (%s context tokens), and all six dry runs.\n' "$context"
run_condition few-shot dry-run
run_condition zero-shot dry-run

if (( preflight_only )); then
  printf '\nPreflight complete. No model requests were sent.\n'
  exit 0
fi

if [[ -z ${OPENAI_API_KEY:-} || -z ${OPENROUTER_API_KEY:-} ]]; then
  printf 'Set OPENAI_API_KEY and OPENROUTER_API_KEY before live runs.\n' >&2
  exit 1
fi

# Score local arms first, before either external provider can spend money.
run_condition few-shot d3-only
run_condition zero-shot d3-only
run_condition few-shot combined
run_condition zero-shot combined

printf '\nBoth conditions completed. Pin the exact completed manifest paths printed above in report/selection.yaml.\n'
printf 'Combined outputs: outputs/comparison/sharks/%s/title_and_desc/{few-shot,zero-shot}/\n' "$expected_hash"
