# GLM-5.2 (unpruned) on 4× DGX Spark — 655,360-token context + MTP k=3 (23.0 tok/s single / 47.9 tok/s at 4 concurrent)

> Serve the **unpruned** GLM-5.2 (744B total / 40B active MoE, 256 experts, QuantTrio int4/int8 mix) with a **655,360-token context window** and **MTP k=3 speculative decoding** across a 4-node NVIDIA DGX Spark (GB10) cluster, using vLLM with decode-context-parallelism (DCP4), fp8 DeepSeek-MLA KV cache, and the B12X sparse-MLA attention backend.

This repository is **self-contained**: image build, source patches, weights, Ray launch, serve command, env vars, and the gotchas that will otherwise eat a day are all here. Hardcoded IPs / users / SSH keys have been genericized with `# EDIT:` markers; substitute your own.

---

## TL;DR

- **Model:** [`QuantTrio/GLM-5.2-Int4-Int8Mix`](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix) — unpruned, 744B/40B MoE, 256 experts, compressed-tensors int4/int8 mix, **~405 GB** on disk.
- **Context:** **655,360 tokens** on 4× GB10 (121 GB unified each). KV cache is sharded 4-way via **decode-context-parallelism (DCP4)**, dtype `fp8_ds_mla`. Boot log reports **`GPU KV cache size: 657,664 tokens`**.
- **Speculative decoding:** **MTP k=3**, measured acceptance length **~3.0** (66% draft-token acceptance overall; per-position matches Zatz's 0.88 / 0.66 / 0.49).
- **Measured decode:** **23.0 tok/s** single-stream, **47.9 tok/s** aggregate at 4 concurrent. Zatz measured decode **flat with depth to 638,976 tokens of context** on this exact recipe (our own deep-depth bench is pending).
- **For:** anyone reproducing max-context GLM-5.2 serving on a 4× DGX Spark cluster. Want max throughput instead of max context? See the companion **200K speed-shape recipe**: [`GLM-5.2-QuantTrio-200K-4x-DGX-Spark`](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark) — same cluster, same checkpoint, 28.8 tok/s single / 60.5 tok/s aggregate at 200K context. Only one shape runs at a time (both need all 4 nodes).

---

## Hardware

- **4× NVIDIA DGX Spark** (GB10, 121 GB unified memory each).
- **Node-to-node RoCE fabric:** 100 G switch, MTU 9000 (jumbo frames) end-to-end. We use a CRS812 switch on subnet `192.168.192.0/24`. NCCL runs RDMA over this interface.
- **~420 GB free disk per node** for the model weights (they are replicated to every node, not shared).
- Docker with the NVIDIA container runtime on all 4 nodes.

The endpoint serves the OpenAI-compatible API on **`:8210`** from the head node.

---

## Quick start

Assumes the image is built and the weights are staged on every node (see [Setup](#setup-detailed)). Scripts live in [`scripts/`](scripts/).

```bash
# 1. Ray cluster up (waits for 4.0/4.0 GPU)
./scripts/launch-ray.sh

# 2. Load the profile
set -a; source scripts/glm52-qt-dcp4-655k.env; set +a

# 3. Serve
./scripts/serve.sh
# -> http://<HEAD_IP>:8210/v1
# -> boot log should read: GPU KV cache size: 657,664 tokens
```

---

## Setup (detailed)

Software stack at a glance:

| Component | What | Source |
|-----------|------|--------|
| vLLM | DCP + shard-draft + global-topk branch | `local-inference-lab/vllm@codex/dcp-globaltopk-sharddraft-defaults-20260622` |
| b12x | sparse-MLA kernels (GB10 / sm_121) | `voipmonitor/b12x@9cd63a72` (build from **source** — PyPI wheel lacks the kernels) |
| Image builder | spark-vllm-docker harness | `eugr/spark-vllm-docker` |
| Attention | `B12X_MLA_SPARSE` (DeepSeek sparse MLA) | b12x |
| Checkpoint | GLM-5.2 int4/int8 mix, compressed-tensors | `QuantTrio/GLM-5.2-Int4-Int8Mix` |
| Patches | 3 in-image source patches (see [`patches/`](patches/)) | this repo |

### Weights

Download once, then rsync to every node over the fast fabric (the weights are read from a **local** path on each node, not a shared mount):

```bash
hf download QuantTrio/GLM-5.2-Int4-Int8Mix \
  --local-dir /var/tmp/models/glm52-int4-int8mix          # ~405 GB

# EDIT: replicate to the other 3 nodes over the RoCE fabric
for n in node2 node3 node4; do
  rsync -a --info=progress2 /var/tmp/models/glm52-int4-int8mix/ \
    "$n":/var/tmp/models/glm52-int4-int8mix/
done
```

### Image / build

1. **Build vLLM with the DCP branch** using the `eugr/spark-vllm-docker` builder:

   ```bash
   VLLM_REPO=https://github.com/local-inference-lab/vllm.git \
   ./build-and-copy.sh \
     --vllm-ref codex/dcp-globaltopk-sharddraft-defaults-20260622 \
     -t vllm-zatz-dcp:probe \
     --tf5
   ```
   (~35–60 min on a Spark.)

2. **Install b12x FROM SOURCE.** The PyPI `b12x` wheel is **missing the sparse-MLA kernels**, so install the pinned commit yourself and bake it into the image:

   ```bash
   git clone https://github.com/voipmonitor/b12x && cd b12x
   git checkout 9cd63a72
   pip install --no-deps --force-reinstall .
   ```

   (Copying the installed `b12x` package directory out of another image that already has it is equivalent — that is in fact what we did.)

3. **Bake the three patches** ([`patches/`](patches/)) into the image. Each is an idempotent, anchored Python patcher — run all three against the installed vLLM. The concrete bake procedure, exactly as we ran it:

   ```bash
   # start a throwaway container from the freshly built image
   docker run -d --name zatz-bake --entrypoint sleep vllm-zatz-dcp:probe infinity

   # (do the b12x source install from step 2 inside this container)

   # apply the three patches — NOTE the -i on docker exec
   for p in fix-mtp-draft-fused-qkv-mapping \
            fix-mla-int32-chunked-prefill \
            fix-dsa-indexer-block-table; do
     docker exec -i zatz-bake python3 - < patches/$p.py
   done

   # commit, explicitly resetting ENTRYPOINT/CMD
   docker commit \
     --change 'ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]' \
     --change 'CMD []' \
     zatz-bake vllm-zatz-dcp:probe
   docker rm -f zatz-bake
   ```

   > **Two docker traps that both bit us:**
   >
   > 1. `docker commit` silently **inherits `--entrypoint` overrides** from the running container. Commit without the `--change` lines above and the image boots into `sleep`, not vLLM. Always reset `ENTRYPOINT`/`CMD` on commit.
   > 2. When piping stdin scripts into a container, use **`docker exec -i`**. Without `-i` the script **silently no-ops** — no error, nothing runs, and you commit an unpatched image.

4. **Distribute the image** to all 4 nodes:

   ```bash
   docker save vllm-zatz-dcp:probe | ssh <node> docker load   # EDIT: node addr/user
   ```

#### Patches

Three source patches are baked into the image. All are idempotent, anchored Python patchers — safe to re-run. See [`patches/README.md`](patches/README.md) for the full write-up; summary:

1. **[`fix-mtp-draft-fused-qkv-mapping.py`](patches/fix-mtp-draft-fused-qkv-mapping.py)** — **the headline, first published fix.** Enables MTP speculative decoding on compressed-tensors (int-quantized) GLM-5.2 checkpoints. The MTP draft builds its quant config from a fresh `CompressedTensorsConfig` with an **empty** `packed_modules_mapping`, so it never learns `fused_qkv_a_proj → [q_a_proj, kv_a_proj_with_mqa]`, silently builds the fused module unquantized (`.weight`, not `.weight_packed`), and `load_weights` then KeyErrors on `model.layers.78.mtp_block.self_attn.kv_a_proj_with_mqa.weight_packed`. The patch seeds the draft's `packed_modules_mapping`.
2. **[`fix-mla-int32-chunked-prefill.py`](patches/fix-mla-int32-chunked-prefill.py)** — (credit Zatz, thread 375416 post 15) fixes garbage output past 4K-token prompts on int-quantized checkpoints: chunked prefill cast activations to the Marlin `int32` weight dtype past the default 4096 chunk size.
3. **[`fix-dsa-indexer-block-table.py`](patches/fix-dsa-indexer-block-table.py)** — the DSA indexer's `expanded_block_table_buffer` is sized from `max_model_len` alone, but the scheduler's block table can be one block wider (MTP spec tokens / scheduler overhang), crashing at concurrency ≥3. Adds `+ 1` block of headroom. Co-discovered independently with forum user ciprianveg (thread 374125 posts 105/107).

### Launch

**Ray (4 nodes).** Use [`scripts/launch-ray.sh`](scripts/launch-ray.sh). It starts a worker container on each of the 3 workers and a head container locally, all on the same image, wired over the RoCE interface with a tiny Ray object store and NCCL channels pinned to 4. Edit the `# EDIT:` block at the top (image tag, model dir, head/worker IPs, interface name, SSH key/user) and run:

```bash
./scripts/launch-ray.sh
# waits until `ray status` shows 4.0/4.0 GPU
```

**Serve.** Load the environment and run the serve wrapper:

```bash
set -a; source scripts/glm52-qt-dcp4-655k.env; set +a   # EDIT: paths/model dir
./scripts/serve.sh
# endpoint: http://<head-ip>:8210/v1
```

The serve command that [`scripts/serve.sh`](scripts/serve.sh) assembles for this profile is, in essence:

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /models --tokenizer /models --served-model-name glm-5.2 \
  --trust-remote-code \
  --quantization compressed-tensors \
  --distributed-executor-backend ray \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 4 --dcp-comm-backend ag_rs \
  --pipeline-parallel-size 1 \
  --attention-backend B12X_MLA_SPARSE \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype fp8_ds_mla --kv-cache-memory-bytes 9000000000 \
  --max-model-len 655360 --max-num-batched-tokens 4096 --max-num-seqs 4 \
  --max-cudagraph-capture-size 32 --async-scheduling \
  --long-prefill-token-threshold 2048 \
  --gpu-memory-utilization 0.88 \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  # ^ REQUIRED for agent use — without these the server runs but silently can't
  #   parse tool calls or separate reasoning from content. See the Troubleshooting section.
  --hf-overrides '{"use_index_cache":true,"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}' \
  --port 8210 --host 0.0.0.0
```

**Required env vars (every node).** These must be in the container environment on **all 4 nodes** ([`scripts/launch-ray.sh`](scripts/launch-ray.sh) sets them):

```bash
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_DCP_SHARD_DRAFT=1
CUTE_DSL_ARCH=sm_121a            # NOT sm_120a — sm_120a gives CUDA error 209 on GB10
TORCH_CUDA_ARCH_LIST=12.1a
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
NCCL_MAX_NCHANNELS=4
NCCL_MIN_NCHANNELS=4
CUDA_DEVICE_MAX_CONNECTIONS=32
```

### Verify

Expect the boot log to report:

```
GPU KV cache size: 657,664 tokens
```

Then confirm tool-calling and reasoning parsing are live: `head -2` of the serve log (or the process cmdline) should show `--reasoning-parser glm45`, `--tool-call-parser glm47`, and `--enable-auto-tool-choice`. A quick functional check is a chat request with a simple `tools` array, confirming a `tool_calls` response comes back.

---

## Benchmarks

**Concurrency** (2026-07-05, 512-token generations, temp 0, shallow context):

| conc | aggregate tok/s | per-stream avg | per-stream min | MTP accept len |
|------|-----------------|----------------|----------------|----------------|
| c1   | 23.0            | 23.0           | 23.0           | 3.01           |
| c2   | 35.1            | 18.0           | 17.5           | 3.03           |
| c3   | 43.4            | 14.6           | 14.5           | 3.01           |
| c4   | 47.9            | 12.5           | 12.0           | 2.91           |
| c5   | 41.3            | 11.8           | 8.3            | 2.99           |
| c6   | 43.4            | 10.8           | 7.2            | 2.97           |

> `max_num_seqs=4`, so at c5/c6 the extra requests **queue** — the aggregate plateau above c4 is by design, not saturation of the GPUs.

**Context-vs-prefill dial.** Pick your shape. Higher DCP buys context at the cost of prefill throughput. (Credit: Zatz, NVIDIA forum thread 375416, posts 12/13/15.)

| config | context | prefill t/s | sustained decode t/s |
|--------|---------|-------------|----------------------|
| no DCP | 143K    | ~716        | 25.5                 |
| DCP2   | 320K    | ~610        | 25.3                 |
| DCP4   | 655K    | ~442        | 23.6                 |

This repo documents the **DCP4 / 655K** endpoint.

---

## Configuration

Key flags, and what each buys you:

- `--decode-context-parallel-size 4` + `--dcp-comm-backend ag_rs` — the KV cache is split 4-way across nodes; this is what makes 655K fit. See the context-vs-prefill dial in [Benchmarks](#benchmarks) for the DCP2 / no-DCP shapes.
- `--kv-cache-dtype fp8_ds_mla` + `--kv-cache-memory-bytes 9000000000` — DeepSeek MLA fp8 KV, hard-capped at ~9 GB, yields the **657,664-token** pool.
- `--attention-backend B12X_MLA_SPARSE` — the sparse-MLA path from b12x.
- `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` — MTP k=3.
- `--hf-overrides … index_topk_pattern …` — the DeepSeek-Sparse-Attention per-layer full/sparse pattern (`F`=full, `S`=sparse). `use_index_cache:true` caches the sparse indexer output.
- `--quantization compressed-tensors` — required for this checkpoint (**not** `modelopt_fp4`, which the RTX defaults assume).
- `--reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice` — **REQUIRED for agent/tool-calling use.** Without them the OpenAI endpoint still serves and answers chat, but tool calls aren't parsed and reasoning tokens bleed into `message.content`. These are gated behind env vars that default OFF in [`scripts/serve.sh`](scripts/serve.sh), so set `REASONING_PARSER=glm45`, `TOOL_CALL_PARSER=glm47`, and `ENABLE_AUTO_TOOL_CHOICE=1` in your env file ([`scripts/glm52-qt-dcp4-655k.env`](scripts/glm52-qt-dcp4-655k.env) ships them set). See [Troubleshooting](#troubleshooting).

---

## Troubleshooting

- **⚠️ Tool-calling & reasoning parsers are opt-in — set them or agents break.** GLM-5.2 needs `--tool-call-parser glm47`, `--reasoning-parser glm45`, and `--enable-auto-tool-choice`. Without them the OpenAI endpoint still serves and answers plain chat, but `tools` / `tool_choice` requests **won't be parsed into tool calls**, and reasoning ("thinking") tokens **bleed into `message.content`** instead of `message.reasoning`. Agent frameworks (Hermes, OpenClaw, OpenCode, etc.) will appear broken even though the server looks healthy. If you drive serving through the recipe/env-file wrapper in this repo, these three are gated behind env vars that default OFF in [`scripts/serve.sh`](scripts/serve.sh), so you must set them in [`scripts/glm52-qt-dcp4-655k.env`](scripts/glm52-qt-dcp4-655k.env) (it ships them set). We shipped a 655K server once without them and had to relaunch — hence this warning. **Verify after boot:** `head -2` of the serve log (or the process cmdline) should show all three flags; a quick functional check is a chat request with a simple `tools` array, confirming a `tool_calls` response comes back.
- **Drop page caches on all nodes before AND during load** (a 60 s loop) — GB10 unified memory stalls big loads in kernel reclaim otherwise.
- **Full Ray purge between image swaps** — `ray stop --force`, then `rm -rf /tmp/ray-vllm-* /dev/shm/ray*` — or you get `ActorHandleNotFoundError` from stale actor handles.
- **`docker commit` inherits `--entrypoint` overrides** — always reset `ENTRYPOINT`/`CMD` on commit (see [Image / build](#image--build) step 3).
- **`docker exec` needs `-i` for stdin scripts** — without it a piped-in patch script silently no-ops and you commit an unpatched image (see [Image / build](#image--build) step 3).
- **`CUTE_DSL_ARCH` must be `sm_121a`** (not `sm_120a`) on GB10 — `sm_120a` errors with CUDA 209.
- **PyPI `b12x` lacks the sparse-MLA kernels** — build from `voipmonitor/b12x@9cd63a72` source.
- **Steady state runs at <1 GB free + a few GB swap by design** — don't co-locate other workloads on these nodes.

---

## Credits & links

This deployment stands on other people's work. Prominently:

- **Zatz** — the 655K DCP4 recipe, the int32 chunked-prefill fix, and the context/prefill dial. NVIDIA developer forum threads **375416** (posts 12/13/15) and **374125** (post 112).
- **CosmicRaisins** — the sm_121 sparse-MLA port, the kernels, and the foundational GLM-5.2-on-Spark work: [github.com/CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10).
- **ciprianveg** — independent co-discovery of the indexer block-table fix (thread 374125 posts 105/107, his `mods/fix-dsa-block-table-dim`).
- **eugr** — the `spark-vllm-docker` build harness.
- **local-inference-lab** and **voipmonitor** — the vLLM and b12x forks.
- **QuantTrio** — the `GLM-5.2-Int4-Int8Mix` checkpoint.

**Contributions from this deployment:** the first published MTP-on-compressed-tensors draft-loading fix; independent co-discovery of the indexer block-table fix; and the first published c1–c6 concurrency numbers for this 655K recipe.

**Related:** [`GLM-5.2-QuantTrio-200K-4x-DGX-Spark`](https://github.com/tonyd2wild/GLM-5.2-QuantTrio-200K-4x-DGX-Spark) — the companion 200K speed-shape recipe on the same cluster.

**License:** Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
