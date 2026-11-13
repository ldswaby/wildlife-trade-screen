# Run d3 locally on an Apple Silicon Mac

The selected d3 arm uses the Q4_K_M GGUF conversion and a loopback llama.cpp server. Allow at least 25 GB of free disk space outside the repository for the model and download overhead.

## 1. Install the tools

```bash
export D3_DIR="$HOME/Models/d3-Q4_K_M"
mkdir -p "$D3_DIR"
df -h "$D3_DIR"
brew install hf cmake
hf --help
```

Set `D3_DIR` to your preferred model directory before downloading.

## 2. Download only the Q4_K_M file

```bash
hf download ggml-org/vllm-d3-GGUF vllm-d3-Q4_K_M.gguf \
  --revision de9670136f5ddd83e677c89ce041d5b9bb3b0dcb \
  --local-dir "$D3_DIR" --dry-run
hf download ggml-org/vllm-d3-GGUF vllm-d3-Q4_K_M.gguf \
  --revision de9670136f5ddd83e677c89ce041d5b9bb3b0dcb \
  --local-dir "$D3_DIR"
ls -lh "$D3_DIR/vllm-d3-Q4_K_M.gguf"
```

Do not download the entire repository: it contains larger variants. Keep the revision, filename, quantization, and server build in the study's run notes. If the pinned revision is unavailable, stop and verify the [GGUF repository](https://huggingface.co/ggml-org/vllm-d3-GGUF) before changing the config or download command.

## 3. Build a d3-capable local server

Build the recorded llama.cpp revision with its bundled ggml and Metal support:

```bash
mkdir -p "$HOME/Developer"
git init "$HOME/Developer/llama.cpp-d3"
git -C "$HOME/Developer/llama.cpp-d3" fetch --depth 1 https://github.com/ggml-org/llama.cpp.git 3e4dfa8f39f904e681737d9f512a6e6338d848aa
git -C "$HOME/Developer/llama.cpp-d3" checkout --detach FETCH_HEAD
cmake -S "$HOME/Developer/llama.cpp-d3" \
  -B "$HOME/Developer/llama.cpp-d3/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_USE_SYSTEM_GGML=OFF \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build "$HOME/Developer/llama.cpp-d3/build" \
  --target llama-server --parallel 8
"$HOME/Developer/llama.cpp-d3/build/bin/llama-server" --version
git -C "$HOME/Developer/llama.cpp-d3" rev-parse HEAD
```

## 4. Start the local server

In the same Terminal window:

```bash
"$HOME/Developer/llama.cpp-d3/build/bin/llama-server" \
  -m "$D3_DIR/vllm-d3-Q4_K_M.gguf" \
  --alias 'ggml-org/vllm-d3-GGUF:Q4_K_M' \
  --host 127.0.0.1 --port 8080 \
  -c 8192 -b 8192 -ub 8192 \
  -np 1 --no-cont-batching
```

Leave the server running. In another terminal, verify readiness with
`curl -fsS http://127.0.0.1:8080/health`. The factorial launcher checks the
model, slot count and context size before live scoring. For multiple d3 slots,
allow at least 8,192 context tokens per slot and enable continuous batching.
Check memory usage on your hardware before increasing concurrency.

Continue with the [campaign instructions](../README.md#4-run-a-complete-campaign).
