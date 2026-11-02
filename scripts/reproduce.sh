#!/usr/bin/env bash
# Rebuild the README results: fetch the models, run both phases of the matrix, render.
#
#   LLAMA_SERVER=~/llama.cpp/build/bin/llama-server OLLAMA=~/ollama/bin/ollama scripts/reproduce.sh
#
# Probes run on cores 8-15; timed sessions take all 16 cores, one setup at a time,
# under an exclusive lock on $BENCH_LOCK once the load average is below 3.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LLAMA_SERVER:?set LLAMA_SERVER to a llama-server binary}"
: "${OLLAMA:?set OLLAMA to an ollama binary}"
export MODELS=${MODELS:-$HOME/models}
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/.ollama/models}
BENCH_LOCK=${BENCH_LOCK:-/tmp/rigcheck-bench.lock}

get() { uvx --from huggingface_hub hf download "$1" "$2" --local-dir "$MODELS/${1#*/}"; }
get unsloth/Qwen3.5-4B-GGUF Qwen3.5-4B-Q4_K_M.gguf
get unsloth/gemma-4-E4B-it-GGUF gemma-4-E4B-it-Q4_K_M.gguf
get ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-MXFP4.gguf
get ggml-org/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-Q4_K_M.gguf

OLLAMA_HOST=127.0.0.1:8091 "$OLLAMA" serve >/dev/null 2>&1 &
pid=$!
sleep 5
for m in qwen3.5:4b gemma4:e4b gpt-oss:20b qwen3.6:35b; do OLLAMA_HOST=127.0.0.1:8091 "$OLLAMA" pull "$m"; done
kill "$pid"
wait "$pid" || true

uv run scripts/matrix.py --phase probes --engines llama.cpp ollama-64k --cpus 8-15 --threads 8 --trials 10 --yield-to "$BENCH_LOCK"
uv run scripts/matrix.py --phase probes --engines ollama --cpus 8-15 --threads 8 --trials 6 --yield-to "$BENCH_LOCK"
uv run scripts/matrix.py --phase probes --engines llama.cpp --models qwen3.6-35b-a3b --agents codex qwen-code minimal \
    --cpus 8-15 --threads 8 --trials 10 --yield-to "$BENCH_LOCK"
uv run scripts/matrix.py --phase timing --cpus 0-15 --threads 16 --sessions 5 --lock "$BENCH_LOCK"
uv run rigcheck report results/*.json --agent opencode --svg docs/results.svg --md docs/results.md --title "rigcheck: OpenCode on a 16-core Neoverse-V2, CPU only"
