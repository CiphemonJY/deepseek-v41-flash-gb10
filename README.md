# Running DeepSeek-V4.1-Flash on 4× 128 GB unified-memory nodes (GB10 / DGX Spark)

Field notes and working code for getting [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
onto a small cluster of 128 GB unified-memory machines at TP=4.

**Status: partial.** Weight conversion, sharding, FP4 kernel validation and transformers
config registration all work and are covered here. vLLM cannot yet *serve* the model — four
features are genuinely absent from its DeepSeek-V4 implementation. This repo documents exactly
which four, so nobody else has to rediscover it.

Everything below was measured on real hardware, not estimated. Numbers that are estimates are
labelled as such.

---

## TL;DR findings

1. **The model does not fit at TP=4 with engram resident.** Backbone 74.5 GiB + engram 47.2 GiB
   = 121.7 GiB per rank, against ~117 GiB usable on a 128 GB node. The engram tables must live
   on disk. They are pure lookup, so this costs essentially nothing.
2. **`mmap` of the fp8 engram table is free.** `safe_open(...).get_tensor()` on a 22.9 GiB fp8
   tensor leaves RSS unchanged at 0.49 GiB, and `F.embedding` gathers straight out of it.
3. **`with torch.device("cuda")` allocates at *construction*.** A loader that swaps tensors for
   mmap views afterwards is too late — the 47.2 GiB is already on the device. Patch
   `__init__` to build on `meta`.
4. **On unified memory, a CUDA overcommit is a *system* overcommit.** It does not raise a tidy
   Python OOM; it starves the kernel, triggers the global OOM-killer against unrelated
   services, and can hard-reboot the box. Always
   `torch.cuda.set_per_process_memory_fraction(~0.8)` first. (Learned the hard way.)
5. **The reference `convert.py` holds every rank's shard in host RAM at once.** A per-rank
   streaming rewrite drops peak to ~28 GiB. Included here.
6. **`requirements.txt` pins `tilelang==0.1.8`, which may break your build.** A vLLM image
   shipping tilelang 0.1.12 fails with
   `AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'` if you downgrade
   to 0.1.8. Use whatever your image already has.
7. **The only V4.1-new *weights* are engram.** Of 96,085 tensors, the only new name
   shapes are 6, across 2 layers. CED, the hierarchical indexer and the DSpark MoE add no
   parameters — they are config-driven compute changes.
8. **V4.1 is an increment on V4, not a new architecture.** V4-Flash already ships `hc_mult`,
   `hc_sinkhorn_iters`, `index_*`, `o_groups`, `o_lora_rank`, `compress_ratios`, `dspark_*`,
   `expert_dtype: fp4`, `scoring_func: sqrtsoftplus` and `num_hash_layers`.

---

## Model shape

From the published shard index (510,296,708,312 bytes = 475.3 GiB, 96,085 tensors):

| component | size | note |
|---|---|---|
| backbone + vision (shards 1–46) | 307 GB = **286 GiB** | 552B params, FP8 dense / FP4 experts |
| engram tables (shards 47, 48) | 203 GB = **189 GiB** | two 101.5 GB shards |
| **total** | **510 GB = 475.3 GiB** | 763.2B params |

Those two outsized shards are the **Engram conditional memory** — the model card's "196B
parameters, sparsely accessed via token-based lookup". 384,006,168 × 256 rows at fp8 lands
almost exactly on 101.5 GB each.

Config geometry, against V4-Flash for comparison:

| | V4-Flash | V4.1-Flash |
|---|---|---|
| layers | 43 | 40 |
| hidden | 4096 | **5120** |
| `moe_intermediate_size` | 2048 | **2304** |
| routed experts | 256 | **384** |
| activated per token | 6 (+1 shared) | 6 (+1 shared) |
| **params per expert** | ~25.2M | **~35.4M** |
| `index_n_heads` | 64 | **32** |
| backbone / activated | 284B / 13B | 552B / 8B prefill, 16B decode |
| `layer_types` | compressed_sparse, heavily_compressed, sliding | compressed_sparse, heavily_compressed |
| `mlp_layer_types` | hash_moe, moe | hash_moe, moe |

Note V4.1's experts are individually **larger**, not smaller. There are simply more of them
with the same 6 activated, which is why activated decode params rise 13B → 16B.

`mlp_layer_types` containing **`hash_moe`** in *both* models is a useful signal: engram is the
evolution of V4's hash layers, so V4's `num_hash_layers` plumbing is the natural place to
attach it.

---

## Memory arithmetic at TP=4

Per-rank, from the actual converted shards:

```
model{i}-mp4.safetensors    79,983,574,400 B  =  74.5 GiB   resident
engram{i}-mp4.safetensors   50,689,508,688 B  =  47.2 GiB   mmap from disk
                                                ---------
                                     total     121.7 GiB    > ~117 GiB usable
```

So engram-on-disk is not an optimisation, it is the thing that makes TP=4 possible at all.
With it on disk you get **74.5 GiB resident, ~42 GiB free** per rank for KV and activations.

The resident figure is ~3 GiB above a naive `286 GiB / 4`, because 1,288 tensors are
replicated across all ranks and `wo_a` is dequantised to bf16 during conversion.

**Why the lookup is cheap:** a decode step touches
`(engram_max_ngram_size - 1) × engram_n_heads = 3 × 8 = 24` rows per engram layer, so 48 rows
total — about 12 KB of a 189 GiB table. At ~50 tok/s that is ~2,600 random 256-byte reads per
second, trivial for NVMe. The table is row-sharded and hash ids are global, so each rank masks
foreign indices to zero and `all_reduce`s — the cross-rank exchange already exists in the
reference implementation.

---

## Step 1 — verify the FP4 kernels run on your hardware

Do this **before** downloading 510 GB. The repo ships a self-test that exercises the real
dense-fp8 / MoE-fp4 tilelang kernels with uninitialised weights:

```bash
cd inference && python model.py
```

On GB10 (SM 12.1) with torch 2.13.0+cu130 and **tilelang 0.1.12** this exits 0. With the
pinned tilelang 0.1.8 it fails inside tilelang's own pre-lower pass:

```
AttributeError: '_NestedLoopCheckVisitor' object has no attribute '_inst'
```

That is a Python/C++ version skew against the bundled TVM, not a hardware verdict.

This matters a lot: `--expert-dtype fp8` routes experts through `cast_e2m1fn_to_e4m3fn`, which
roughly **doubles** the dominant memory term. If FP4 does not work on your hardware, the
fallback takes the backbone from ~286 GiB to ~520 GiB and nothing fits at any TP.

> Check the exit code directly. `python model.py | tail -40` reports tail's status, not
> python's, and will happily show you a traceback next to `exit 0`.

---

## Step 2 — convert with per-rank streaming

[`tools/convert_streaming.py`](tools/convert_streaming.py) is a drop-in replacement for the
reference `inference/convert.py`.

The reference builds `state_dicts = [{} for _ in range(mp)]` and fills **all** `mp` shards
while walking the inputs, writing nothing until the end. Three changes:

1. **Rank is the outer loop** — `mp` passes over the inputs, holding only one rank's share.
2. **Engram is split out** to `engram{i}-mp{mp}.safetensors`, so the backbone file stays
   74.5 GiB and the tables can be mmap'd at serve time.
3. **Engram uses `get_slice()`, not `get_tensor()`** — `layers.N.engram.embed.weight` is a
   single **91.6 GiB** tensor; slicing lazily reads only this rank's ~22.9 GiB.

```bash
python tools/convert_streaming.py \
  --hf-ckpt-path /path/to/DeepSeek-V4.1-Flash \
  --save-path    /path/to/DeepSeek-V4.1-Flash-TP4 \
  --model-parallel 4 --expert-dtype fp4 \
  --tokenizer-path /path/to/DeepSeek-V4.1-Flash
# one rank at a time, resumable or spread across machines:
#   --ranks 0,1
```

Measured: **9 minutes, peak RSS 28 GiB**, on a node with the inputs on local NVMe.

In fairness: safetensors returns mmap-backed views, so the reference converter's peak is lower
than its structure suggests and may well fit on a large-RAM box. The per-rank loop costs
nothing and removes the question; the `get_slice` change and the engram split are the parts
that genuinely matter.

Verify the output — expert ranges should be disjoint and complete:

```
model0: experts   0..95    mtp  0..31    25190 tensors   dtypes include F4
model1: experts  96..191   mtp 32..63
model2: experts 192..287   mtp 64..95
model3: experts 288..383   mtp 96..127
engram{0..3}: 4 tensors; layer1 rows=96,001,542  layer14 rows=96,004,171
```

`4 × 96,004,171 = 384,016,684` against a true `384,016,682`, so the ceil-division pad of 2 rows
is exercised for real — not dead code.

---

## Step 3 — load the split checkpoint

[`tools/load_split.py`](tools/load_split.py) replaces the reference's single
`load_model(...)` call. Two functions must be called **in this order**, and the order is the
entire point:

```python
from load_split import guard_cuda_memory, patch_engram_for_mmap, load_split
import model as model_mod

guard_cuda_memory(0.82)             # BEFORE anything touches CUDA
patch_engram_for_mmap(model_mod)    # BEFORE Transformer(...) is constructed
with torch.device("cuda"):
    model = Transformer(args, tokenizer)
load_split(model, ckpt_path, rank, world_size)
```

* `patch_engram_for_mmap` builds the engram tables on the **`meta` device (0 bytes)** while
  keeping them in `state_dict()`, so `load_model(strict=False)` still reports them missing and
  `load_split` can verify the split is exactly as expected before substituting mmap tensors.
  It fails closed: if the set of missing backbone keys is not exactly the set of engram keys,
  it raises rather than silently serving a partly-initialised model.
* `guard_cuda_memory` caps the process via `set_per_process_memory_fraction`. **Do not skip
  this.** Without it, a 121.7 GiB allocation against a 121.7 GiB unified pool does not fail
  politely — it starves the kernel, the global OOM-killer reaps whatever it likes, and the
  machine can reboot. Unified memory means a CUDA overcommit is a system overcommit.

---

## Step 4 — register `deepseek_v41` with transformers

[`config/dsv41_config.py`](config/dsv41_config.py). **No released transformers knows this
architecture** — verified against 5.16.1 and 5.17.0, both of which ship
`deepseek_v2/v3/v32/v4` and no v41. Without this you get:

```
ValueError: The checkpoint you are trying to load has model type `deepseek_v41`
but Transformers does not recognize this architecture.
```

The shim is small because `DeepseekV4Config` already declares most of the machinery, and
`PreTrainedConfig` keeps undeclared kwargs as attributes. Three real differences are handled:

**a) V4.1 nests `text_config` / `vision_config`** (each with its own declared `model_type`)
where V4-Flash was flat with `vision_*` prefixed keys.

**b) `compress_ratios` is not a rename of V4's `compress_rates`.** transformers types the
latter `dict | None`; V4.1's is a per-**layer** list of length `n_layers + 3` (43 for 40
layers, covering the MTP layers). Feeding one into the other trips strict dataclass
validation. Keep them separate.

**c) The rope split must be flattened.** `DeepseekV4Config` synthesises `rope_parameters` as
two named sub-configs, `main` and `compress` (the latter carrying `rope_theta: 160000`). vLLM
only treats a nested rope dict as nested when **every key is a layer type**:

```python
is_rope_parameters_nested(rp) = set(rp) <= ALLOWED_LAYER_TYPES
# ('full_attention', 'sliding_attention', ..., 'compressed_sparse_attention', ...)
```

`{main, compress}` is not a subset, so vLLM wraps it as `{"": {main:…, compress:…}}` and then
dies on `rp["rope_type"]`. vLLM's own loader already flattens this for `deepseek_v4` — confirmed
by running `get_config()` against a V4-Flash checkpoint, which yields a single flat yarn dict —
but that normalisation does not fire for `deepseek_v41`. The shim reproduces its output
exactly, keeping **main's** `rope_theta` (10000), not compress's:

```python
{'rope_type': 'yarn', 'type': 'yarn', 'factor': 16, 'beta_fast': 32, 'beta_slow': 1,
 'original_max_position_embeddings': 65536, 'partial_rotary_factor': 0.125,
 'attention_factor': 1.0, 'rope_theta': 10000}
```

`compress_rope_theta` stays reachable at 160000, which is where the model code reads it.

With that in place, `vllm.config.ModelConfig` resolves, and an uncapped `max_model_len`
derives **1,048,576** (65536 × 16). [`tests/test_config_stability.py`](tests/test_config_stability.py)
covers 31 assertions — idempotent registration, determinism over repeated parses, the rope
shape, resolution at several context lengths, and structural field integrity.

---

## What is still missing to actually serve it

Register the architecture and vLLM gets past the config and stops at its model registry. The
V4 package is substantial and already implements most of what V4.1 needs:

| feature | present in vLLM's DeepSeek-V4 | |
|---|---|---|
| `compress_ratio` (CSA) | 221 references | ✅ |
| `indexer` | 141 | ✅ |
| `hc_mult` + `sinkhorn` (mHC) | 109 + 20 | ✅ |
| `expert_dtype` (FP4 experts) | 58 | ✅ |
| `swiglu_limit` | 33 | ✅ |
| `o_lora` / `o_groups` | 13 / 1 | ✅ |
| `sqrtsoftplus` | 7 | ✅ |
| `num_hash_layers` / `hash_layer` | 3 / 3 | ✅ hook exists |
| **engram** | **0** | ❌ |
| **`kv_source_layer_ids` / `index_source_layer_ids`** (CED) | **0** | ❌ |
| **`candidate_*`** (hierarchical sparse indexer) | **0** | ❌ |
| **`dspark_n_routed_experts`** (DSpark MoE) | **0** | ❌ |

So four deltas on a mature base — and **only the first touches the weight loader**:

1. **Engram** — 6 tensor shapes across 2 layers (`embed.weight/.scale`, `wkv.weight/.scale`,
   `q_weight`, `k_weight`). The reference module is ~75 lines; attach at the existing
   `hash_moe` layer type.
2. **CED** — decoder layers project global KV from the final encoder hidden states rather than
   their own, driven by `kv_source_layer_ids` / `index_source_layer_ids`. Needs per-layer KV
   aliasing. No new weights.
3. **Hierarchical sparse indexer** — `candidate_source_layer_id`, `candidate_topk_blocks`,
   `candidate_block_size` restrict deeper indexing layers to a candidate pool built by the
   first Full-mode layer. Extends the existing indexer. No new weights.
4. **DSpark routed MoE** — `dspark_n_routed_experts: 128`,
   `dspark_num_experts_per_tok: 3`. The draft model already exists. MTP expert weights are
   already in the checkpoint under `mtp.` names.

Note CED is the reason tensor-parallelism is a better fit than pipeline-parallelism here:
CSA2 shares KV across layers, so under TP every rank holds every layer and that reuse stays
local, whereas a PP stage boundary cuts straight through it.

---

## Expected performance (estimated, not measured)

Derived from an effective ~107 GB/s per node measured on a comparable TP=4 MoE decode
(~39% of the 273 GB/s spec — batch-1 multi-node MoE decode is latency-bound, not
bandwidth-bound). **These are projections; treat them as such.**

Assuming MLA's compressed KV is replicated across ranks rather than sharded:

| context | V4-Flash | V4.1 | delta |
|---|---|---|---|
| short | ~55 tok/s | ~45 | −19% |
| 128K | ~45 | ~43 | −4% |
| 500K | ~29 | **~38** | **+30%** |
| 1M | ~20 | **~33** | **+66%** |
| prefill | ~2,720 tok/s | **~4,400** | **+62%** |

V4.1 reads ~23% more weight per decode token (16B vs 13B activated), so it loses on short
prompts. It wins wherever context is long, because KV drops from ~3.5 KB to 890 B per token.

**Concurrency improves** despite the heavier model: ~34.5 GiB of KV budget per rank at 890 B/token
is ~41.6M tokens, against ~21.4M for V4-Flash at ~3.5 KB — roughly **1.9× more concurrent
tokens**. Under continuous batching, aggregate throughput should improve everywhere except
short-prompt/short-output traffic.

DSpark speculative decoding is deliberately excluded from these numbers — its acceptance rate
is unmeasured, and V4.1 drafts 5-token blocks where V4 uses MTP depth 3.

---

## Gotchas worth repeating

* `exfat` cannot store symlinks, and the HF cache is `blobs/` + `snapshots/` symlinks. Copying
  a cache directory to an exfat volume silently flattens it. Use `tar`.
* Mounting only `snapshots/<rev>/` into a container gives you dangling symlinks and a baffling
  "config.json has no `model_type`". Mount the model root so `blobs/` comes too.
* A multi-rank launcher that does not abort when one worker fails to start will launch the rest
  and leave them spinning in NCCL indefinitely. Check yours, and pause any watchdog while a
  rank is wedged or it will rebuild the broken N−1 group for you.
* `torch.distributed`'s 30-minute default `init_process_group` timeout is what eventually frees
  ranks orphaned by a dead rank 0. Until then they hold their allocation and can thrash hard
  enough to starve `sshd` — a box that answers ping but times out "during banner exchange" is
  memory-starved, not network-broken.

---

## Provenance

Measured on four 128 GB GB10 nodes at TP=4, torch 2.13.0+cu130, tilelang 0.1.12, a vLLM build
carrying DeepSeek-V4 support. Model weights are MIT licensed by DeepSeek-AI. This repo is MIT.

Corrections welcome — particularly from anyone who gets the four deltas implemented.
