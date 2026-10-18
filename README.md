# vLLM DGX Spark Recipes

Ready-to-run starter scripts for serving LLMs on a single **NVIDIA DGX Spark**
(GB10, SM121, 121 GiB unified memory) with [vLLM](https://github.com/vllm-project/vllm).

Each recipe is a self-contained folder with:

- a `starter-*.sh` script that assembles and runs the right `docker run` command
  (foreground, detached with health check + smoke test, or dry-run);
- a `README.md` documenting every env knob, the serving profile, and benchmark numbers.

## Recipe index

Recipes live in a folder named after the exact Hugging Face repo ID of the checkpoint.

| Recipe | Model | Quantization | Notes |
|--------|-------|--------------|-------|
| [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](nvidia/Qwen3.8-Flash-Next-FP4-Local/) | Qwen3.8-Flash-Next (~125B total / 6B active MoE) | NVFP4 (CSF scales) | Requires community b12x container; MTP speculative decoding |

## Requirements

- NVIDIA DGX Spark (GB10, SM121) with 121 GiB unified memory
- Docker with GPU support (`nvidia-container-toolkit`)
- The checkpoint downloaded locally (each recipe documents the expected path)

## Quick start

```bash
cd <recipe-folder>
./starter-*.sh                 # foreground
DETACH=1 ./starter-*.sh        # background + health wait + smoke test
DRY_RUN=1 ./starter-*.sh       # print the docker command without running it
```

Every script exposes its knobs as env vars set before the script name, e.g.:

```bash
MAX_MODEL_LEN=131072 GPU_MEM_UTIL=0.75 PORT=8082 DETACH=1 ./starter-*.sh
```

See each recipe's README for the full variable reference.

## Adding a recipe

1. Create `<org>/<model-name>/` matching the Hugging Face repo ID.
2. Copy a `starter-*.sh` from an existing recipe and adapt the serving profile
   (quantization flags, parsers, speculative decoding, env block).
3. Add a `README.md` with: quick start, why-not-stock-vLLM notes (if any),
   full env var table, benchmark numbers, model summary, requirements.
4. Add a row to the recipe index in this file.

## License

MIT — see [LICENSE](LICENSE). Scripts are provided as-is; use at your own risk.
