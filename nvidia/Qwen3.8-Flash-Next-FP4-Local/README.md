# nvidia/Qwen3.8-Flash-Next-FP4-Local on DGX Spark (GB10)

Starter script for serving [nvidia/Qwen3.8-Flash-Next-FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) on a single NVIDIA DGX Spark (GB10, SM121) with vLLM.

## Quick start

```bash
./starter-qwen38-flash-next-fp4.sh                 # foreground
DETACH=1 ./starter-qwen38-flash-next-fp4.sh        # background + health wait + smoke test
DRY_RUN=1 ./starter-qwen38-flash-next-fp4.sh       # print docker command
```

Stop: `docker rm -f qwen38-flash-next` (or your `CONTAINER_NAME`) — Logs: `docker logs -f qwen38-flash-next`

## Configuration

This checkpoint's NVFP4 scale factors use a CSF (compressed) format that upstream vLLM cannot load. You must use a community container with `--load-format b12x`:

```bash
IMAGE=eugr/spark-vllm-b12x:latest  # default, arm64 for GB10
```

Stock `vllm/vllm-openai` images will fail to load the weights even with the b12x wheel.

### Common

| Var | Default | Description |
|-----|---------|-------------|
| `IMAGE` | `eugr/spark-vllm-b12x:latest` | Container with CSF NVFP4 loader |
| `MODEL_DIR` | `~/models/Qwen3.8-Flash-Next-FP4-Local` | Host path to the checkpoint (mounted read-only at `/model`) |
| `GPU_MEM_UTIL` | `0.89` | Fraction of unified memory for vLLM (keep `<= 0.9`, the OS shares unified memory) |
| `MAX_MODEL_LEN` | `524288` | Context window (yarn rope from native 262144, see below) |
| `MAX_NUM_SEQS` | `7` | Max concurrent sequences |
| `MAX_NUM_BATCHED_TOKENS` | `16384` | Prefill chunk size |
| `PORT` | `8081` | Host port |
| `HOST` | `0.0.0.0` | Bind address |
| `SERVED_NAME` | `qwen38-flash-next-fp4` | OpenAI-compatible model name |
| `SPEC` | `mtp` | Speculative decoding: `mtp` or `none` |
| `SPEC_TOKENS` | `4` | Draft tokens per MTP step |
| `PLE_OFFLOAD` | `1` | 1 = n-gram table on disk, 0 = RAM |
| `CONTAINER_NAME` | `qwen38-flash-next` | Docker container name |
| `DETACH` | `0` | 1 = run in background, wait for health, smoke test |
| `DRY_RUN` | `0` | 1 = print the docker command and exit |
| `EXTRA_ARGS` | | Extra vLLM flags after `--` (word-split on spaces, see below) |
| `EXTRA_ENV` | | Extra `-e KEY=VALUE` env entries (space-separated `KEY=VALUE` pairs) |

Example:
```bash
MAX_MODEL_LEN=131072 GPU_MEM_UTIL=0.75 PORT=8082 DETACH=1 ./starter-qwen38-flash-next-fp4.sh
```

### Context window and rope scaling

The checkpoint natively supports 262144 tokens. When `MAX_MODEL_LEN > 262144`
(the default is 524288), the script auto-injects `--hf-overrides` with a yarn
rope config: `factor = MAX_MODEL_LEN / 262144` (2.0 for the default),
`original_max_position_embeddings=262144`, interleaved mrope.

- `MAX_MODEL_LEN <= 262144` → no override is injected, native config is used.
- Set `HF_OVERRIDES` explicitly to take full control (an explicitly set value
  is always passed through untouched, even if empty).

### Advanced — container env (b12x / AOT)

| Var | Default | Description |
|-----|---------|-------------|
| `VLLM_PLE_TABLE_MEMORY` | derived from `PLE_OFFLOAD` | `disk` or `ram` |
| `B12X_DISK_BACKEND` | `io_uring` | Disk backend for the offloaded n-gram table |
| `CUTE_DSL_ARCH` | `sm_121a` | CuTe DSL target arch for GB10 |
| `SAFETENSORS_FAST_GPU` | `1` | Fast safetensor loading |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | Worker spawn method |
| `VLLM_SSM_CONV_STATE_LAYOUT` | `DS` | SSM conv-state layout |
| `VLLM_USE_AOT_COMPILE` | `1` | AOT-compile kernels |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | `1` | Reuse the mega AOT artifact |
| `VLLM_USE_V2_MODEL_RUNNER` | `1` | V2 model runner |
| `B12X_POLICY_MODE` | `auto` | b12x kernel policy mode |
| `VLLM_MXFP8_LM_HEAD` | `1` | MXFP8 LM head |
| `VLLM_B12X_MOE_FP4_LAYER_MAX_INPUT_SCALE` | `w13` | MoE FP4 input-scale source |
| `VLLM_B12X_MOE_FP4_FORCE_A16` | `1` | Force FP16 activations in FP4 MoE layers |
| `B12X_W4A16_A4_PREFILL` | `1` | 4-bit weights / 4-bit act prefill path |
| `OMP_NUM_THREADS` | `2` | OpenMP threads (leave headroom for CPU-bound work) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | CUDA allocator config |

## First boot and caches

- Loading ~96 GiB of weights plus AOT kernel compilation can take a long time.
  `DETACH=1` waits up to **1200 s (20 min)** for `/v1/models` to answer, then
  runs a 32-token smoke test. Subsequent starts are much faster.
- These host directories are mounted into the container and persist across
  restarts, so compiled artifacts and indexes are reused:
  `~/.cache/huggingface`, `~/.cache/vllm`, `~/.cache/flashinfer`,
  `~/.cache/triton`, `~/.cache/b12x`.
- The host needs `curl` and `python3` for the detached health check / smoke test.

## Performance (DGX Spark GB10, 121 GiB unified memory)

Benchmarked with AIPerf via AI gateway.
Test: 128 input tokens, 512 output tokens, 40 requests, MTP-4 speculative decoding.

| Concurrency | Active Decode (agg) | Per User | TTFT | Request Latency | p50 Decode (agg) |
|-------------|---------------------|----------|------|-----------------|------------------|
| 1 | 41.2 tok/s | 41.2 tok/s | 348 ms | 12.7 s | 40.5 tok/s |
| 2 | 58.4 tok/s | 30.1 tok/s | 525 ms | 17.4 s | 60.1 tok/s |
| 4 | 84.8 tok/s | 22.2 tok/s | 682 ms | 23.7 s | 88.7 tok/s |

Notes:
- Aggregate scales ~linearly (41 → 58 → 85 tok/s); the GB10 is not yet saturated at c=4.
- Per-user throughput decreases as expected (shared GPU).
- TTFT and latency increase moderately — no queueing bottleneck.
- Min/max outliers (0 / 6733 tok/s) are from failed/short responses; use p50.

### Concurrency sweep (512 in / 512 out)

AIPerf run: 512 input tokens, 512 output tokens, 40 requests per concurrency
level (120 total, 0 errors), MTP-4 speculative decoding.

| Concurrency | Active Decode (agg) | Per User | TTFT | Request Latency | p50 Decode (agg) |
|-------------|---------------------|----------|------|-----------------|------------------|
| 1 | 44.8 tok/s | 44.5 tok/s | 472 ms | 11.9 s | 44.2 tok/s |
| 2 | 60.7 tok/s | 32.0 tok/s | 657 ms | 16.7 s | 60.4 tok/s |
| 4 | 86.6 tok/s | 23.3 tok/s | 890 ms | 22.8 s | 89.7 tok/s |

Notes:
- Aggregate decode scaling (45 → 61 → 87 tok/s) matches the 128-ISL sweep
  (41 → 58 → 85) — prefill size does not affect decode-phase scaling.
- TTFT is ~1.4× higher than the 128-ISL run despite 4× longer prompts;
  prefill stays above 1,100 tok/s even at c=4 (512 tokens / 0.47 s).
- ITL p50: 22.1 / 32.3 / 43.4 ms for c=1/2/4.

### Long context (500K tokens)

AIPerf run: 3 requests × 499,945 input tokens, 32 output tokens, concurrency 1,
3 shuffled prompts so the prefix cache cannot serve requests 2/3 (cold prefill each time).

| Metric | Value |
|--------|-------|
| Requests | 3/3 completed, 0 errors |
| Prefill throughput (avg) | **1,652.7 tok/s** (min 1,651.9 / max 1,659.5) |
| TTFT (avg) | **302.1 s** (~5.0 min per 500K-token prompt) |
| Request latency (avg) | 302.5 s |
| Inter-token latency (p50) | 10.1 ms (~99 tok/s/user decode at 500K context) |
| Total wall time | 907 s (3 serial requests) |

Notes:
- Prefill variance is ±4 tok/s across all 3 requests — no thermal throttling
  or fragmentation; request 3 is as fast as request 1.
- Decode at 500K context shows no KV-cache read penalty (~10 ms ITL).
- Decode throughput at OSL=32 is directional only (~7 decode steps; MTP bursts
  make max values meaningless — use ITL p50).

## Model

- **Architecture**: Hybrid Gated DeltaNet + Qwen Sparse Attention + MoE (512 experts) + n-gram embedding (PLE) + native MTP head
- **Size**: ~125B total (6B active) + 51B n-gram + 4B MTP, ~96 GiB on disk
- **Quantization**: NVFP4 QAD (modelopt_mixed, CSF scales)
- **KV cache**: FP8, block-size 16
- **Speculative decoding**: MTP, 4 draft tokens
- **Reasoning**: `--reasoning-parser qwen3`, toggle per-request with `chat_template_kwargs {"enable_thinking": true/false}`
- **Tool calling**: `--tool-call-parser qwen3_coder`

## Requirements

- NVIDIA DGX Spark (GB10, SM121) with 121 GiB unified memory
- Docker with GPU support
- Model at `~/models/Qwen3.8-Flash-Next-FP4-Local` (or set `MODEL_DIR=`)

