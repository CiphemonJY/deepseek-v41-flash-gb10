# Running DeepSeek-V4.1-Flash on 4× 128 GB unified-memory nodes (GB10 / DGX Spark)

Field notes and working code for getting [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
onto a small cluster of 128 GB unified-memory machines at TP=4.

**Status (2026-09-11): SERVING on vLLM, TP=4 — eager + DSpark k=5 (30-39 tok/s single-stream), vision + tools. CUDA graphs: no usable configuration on this fleet after 12 boots / 2 lineages / 3 NCCL envs (see the graphs section).**

| stage | state |
|---|---|
| FP4 MoE kernels on SM 12.1 | ✅ verified (`model.py` self-test, exit 0) |
| TP=4 conversion + sharding | ✅ verified (9 min, expert ranges disjoint, `F4` preserved) |
| engram mmap lookup | ✅ verified (22.9 GiB fp8 table at 0.49 GiB RSS) |
| `deepseek_v41` config through vLLM's `ModelConfig` | ✅ **end-to-end, 31 assertions** |
| reference-implementation model **load** | ✅ with `load_split.py` (per-tensor streaming, engram on `meta`) |
| reference-implementation **generation** | OK - verified, correct text AND correct multimodal answers |
| vLLM config + architecture registration | superseded — the `deepseek_v4_1` tree ships its own; the shim was only needed on the V4-only build |
| vLLM **serving** | ✅ **eager + DSpark k=5, vision + tool calling, 131K ctx, ~30-35 tok/s** — `launch/`, `build/`, `gate/` (Served section). CUDA graphs: garbled on this image (bisected) |

The config path is genuinely end-to-end: `AutoConfig` parses the real checkpoint and
`vllm.config.ModelConfig` resolves it at four context lengths. Everything downstream of that
is where the remaining work is, and this repo is explicit about which parts are verified and
which are written-but-unproven.

Everything below was measured on real hardware, not estimated. Numbers that are estimates are
labelled as such.

---

## Served — how V4.1 actually runs on vLLM here (2026-09-11)

The working path is the one in [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark)
(vLLM branch `dsv41-feat` + seven SM12x/engram-on-disk patches), adapted to a fleet with the full checkpoint local on every rank.

**Image** (`build/`): `vllm/vllm-openai:deepseekv41-flash-0909-arm64` -> `+ FlashInfer 0.7.0rc1` (0.6.18 lacks the SM120
sparse-MLA decode for V4.1's top-k 1152; `Dockerfile.fi07`) -> `+ prewarmed sparse_mla_sm120 / mxfp8 kernels` (`Dockerfile.v41`,
`prewarm5.py`) -> `+ the dsv41-feat python tree COPIED over the package` (`Dockerfile.branch`). The 0909 image's compiled
`_C_stable_libtorch` is kept: it loads on GB10 and covers every custom op the V4.1 NVIDIA path references (verified by scan).
**Why the tree swap:** the 0909 python tree is a *different revision* from `dsv41-feat` — its `engram.py` carries a DP-engram
API (`gather_engram_hashes`) that the branch lacks, so the branch's whole-file patches cannot sit on it (`ImportError`).
6 of the 7 patch files are byte-identical to 0909's originals + the reference diffs; only `engram.py` diverges.

**Patches:** the reference's seven files, bind-mounted per its `patch/mounts.txt`. **Verify their md5s against the
reference's own table (`patch/README.md`) before every launch** — see the CRLF gotcha below.

**Launch** (`launch/launch41.sh <rank>`, `launch/boot41.sh`): the reference's `dsv41-tp4.sh` shape, plus a fail-closed
**pre-launch gate** (md5 vs reference table, CR-byte scan, and an in-container import of the model package asserting each
patch's marker symbol is live), **host-side `docker logs -f` capture** (an engine log must survive a node reboot), and a
**kill-switch** (`docker kill` at MemAvailable < 2 GiB — a contained failure instead of the hang-guard panic). `gate/check_patches.sh`
is the standalone no-GPU version of the gate. Set `HEAD_IP`, `RANK{0..3}_IP`, `RANK{0..3}_SSH`, `NCCL_IB_HCA`, `FABRIC_IFACES`,
`FABRIC_IFACE0`, `DSV41_HOME`, `SERVED_NAMES` for your fleet.

**Serving config (final):** `GMU=0.80 MAXLEN=131072 SEQS=8 EAGER=1 SPEC=dspark SPEC_K=5 TEXT_ONLY=0 PARSERS=1` with
`--limit-mm-per-prompt {"image":4} --mm-processor-cache-gb 1`. Measured: 48 shards load in ~45 s from local NVMe;
~9 GiB free at steady state; **30-39 tok/s** single-stream with DSpark across restores (14.9 without); the reference's vision + tool-calling
suite 7/7; a 4-way concurrent garble check clean. `GMU=0.72` fails cleanly with `No available memory for the cache blocks` (~79 GiB of weights leave no KV).

**Bisected — CUDA graphs are broken on this image; DSpark is fine.** Graphs ON + DSpark OFF: boots, HTTP 200, output is
garbage (U+FFFD). Graphs ON + DSpark ON (the reference's boot 10): first request dies with `CUDA error: an illegal memory
access` in `fused_moe/runner/moe_runner.py:_maybe_reduce_final_output` — DSpark's rejection sampler consuming the corrupted
memory. DSpark ON + graphs OFF: clean, 2.2x faster than plain eager. **Rebuilding `_C_stable_libtorch` from the branch's csrc did NOT fix it** (maintenance window, 2026-09-11): the reference's
`build_stable_ext.sh` takes **6 minutes** on a GB10 (-j16, `NVCC_THREADS=2`, min 71 GiB free, cgroup-capped — not hours), and
this tree's `CUDA_SUPPORTED_ARCHS` tops out at 12.0, so `TORCH_CUDA_ARCH_LIST=12.1a` becomes `sm_120` + `sm_120f` (the reference's
"sm_121a" is the same thing). With the rebuilt extension, graphs ON still yields empty/garbage text on every request
(`/v1/completions` returns `''` for 40 generated tokens; chat `content: null`). So the corruption is not in that extension.
Remaining suspects, untested: the exact nightly base they used (`nightly-8a728663`) vs the 0909 image's other libraries
(torch/NCCL/FlashInfer build), and the graph-replayed sparse-MLA/collective path on this fabric. Eager + DSpark is the
stable configuration here.

### CUDA graphs on this fleet: investigated, no usable configuration (2026-09-11, 12 boots)

Two full image lineages were built and tested: the 0909 tag + branch tree (`Dockerfile.branch`) and the reference's exact
lineage (`build/nightly_chain.sh`: `nightly-8a728663` + branch tree + `_C_stable_libtorch` rebuilt against its torch via
`ext_build_inner.sh` in 6 minutes + FlashInfer 0.7.0rc1 + prewarms). Both load, pass the gate, and serve correctly in eager.
With CUDA graphs on (`EAGER=0`), **11 of 12 boots produced garbage from the first token** (engine up, HTTP 200, `�care…`),
across three NCCL environments (the previous serve's dual-rail/LL128 env, the same minus `NCCL_PROTO=LL128`, and the reference's
plain env with a 4-channel cap). The single clean graph boot was the minimal config — no DSpark, `--language-model-only`, no
parsers, LL128 dropped — at 28.8 tok/s cold, observed once and not reproduced; every combination that adds DSpark, vision, or
the parsers garbled, so none of those is the discriminator. That minimal config is slower than eager + DSpark (30-39 tok/s)
and drops features, so **graphs are off in production here**. Forced `NCCL_PROTO=LL128` is the best-supported suspect (all
three lineages garbled with it; the one clean boot was without it) but not a certified cause. `launch41.sh` exposes
`NCCL_DROP="PROTO ..."` / `NCCL_SET="K=V ..."` / `NCCL_ENV_MODE=ref` for further bisecting; note the reference's plain env
without a channel cap costs ~7 GiB of load-time headroom on this fleet and gets killed by the memguard. Untested: host-level
differences from the reference (driver/firmware/`90-spark-hang-guard` sysctls) and `VLLM_USE_BREAKABLE_CUDAGRAPH`.

**Production config:** `EAGER=1 SPEC=dspark SPEC_K=5 TEXT_ONLY=0 PARSERS=1 GMU=0.80 MAXLEN=131072 NCCL_DROP=PROTO` on the
0909-lineage image — 30-39 tok/s single-stream, 7/7 vision + tools, 4-way concurrent garble check clean.

### Second investigation (council + red-team loop, 2026-09-11 evening): the defect, named

Six more graph boots with per-boot instrumentation (`tools/garble_gate.py` greedy token-level gate against an eager
baseline, `tools/battery.py`, `tools/xid_since.sh`, `tools/postboot.sh`; launcher knobs `LOG_LEVEL`, `ALLOC_CONF`,
`EXTRA_ENV`, `CACHE_TAG`, `mounts.extra.txt`) established:

* **The minimal graph config is real:** `EAGER=0`, no DSpark, `--language-model-only`, no parsers, LL128 dropped — clean
  on replication under `VLLM_LOGGING_LEVEL=DEBUG` (2/2), 2000-token prompt and 6 concurrent streams clean, zero Xids.
* **The full config fails on the first request in 4/4 boots.** In three of them (19:03Z, 19:34Z, 19:54Z) the head log
  carries `tvm.error.InternalError: [MXFP8 SM120 gemm Runner] Failed to initialize cutlass MXFP8 gemm on sm120` and
  `Failed to initialize the TMA descriptor 700` (a sticky illegal-address error surfacing at a detection point); in the
  19:54Z boot the first response also carried `Out of range float`, i.e. a NaN first forward (the other boots did not log it); the 19:13Z boot faulted with illegal-access errors
  without those two lines. Every event put `Xid 31 ... ACCESS_TYPE_VIRT_READ` on **all four ranks within the same
  second**; in one event all four faulted at the same virtual address, in another the addresses differed and one rank
  reported `FAULT_PTE` rather than `FAULT_PDE`. The same-second/all-ranks pattern points at deterministic software; the
  VA evidence is weaker than a single identical address would be. **What triggers it:** the minimal config is clean;
  every boot that added *two or more* of {DSpark, vision wrapper, parsers} faulted, including {vision, parsers} without
  DSpark. No boot added exactly one factor, so "DSpark alone" or "vision alone" is **not established**.
* **Falsified, each by a single-variable boot:** gmu headroom (0.72 leaves no KV pool; the full config needs ≥0.78),
  `PYTORCH_CUDA_ALLOC_CONF=backend:native` (same fault, louder), `--max-num-batched-tokens 4096` + no mm cache,
  `--no-enable-prefix-caching`, a strong-reference patch to `compilation/breakable_cudagraph.py` (`tools/mk_strongref.py`,
  mountable via `mounts.extra.txt`), the NCCL environment (three variants), and the image lineage (two).
* **Tree provenance:** upstream deleted `dsv41-feat` on 2026-09-11; successors are `dsv41-optimized` and
  `feat/dsv41-swa-bounded-replay` (the latter is a prefix-cache replay fix for the SWA window, not a CUDA-graph fix). The
  reference's exact snapshot is unrecorded, so "same tree as the reference" cannot be verified.

Production stays on **eager + DSpark k=5 + vision + tools** (30-39 tok/s single-stream across restores: 8-second,
n=1-2 samples with co-tenant load uncontrolled and no throttling signature in `nvidia-smi`, so read it as ~30-35 with noise). The
graphs+DSpark config is an open upstream defect on this tree/hardware combination; the evidence above is what an issue
needs.

### Required environment for the published scripts

`launch/launch41.sh` and `launch/boot41.sh` refuse to run until these are set (`${VAR:?}`): `HEAD_IP` (rank-0 fabric IP),
`RANK0_IP`..`RANK3_IP`, `RANK0_SSH`..`RANK3_SSH` (ssh targets used by `boot41.sh`), `NCCL_IB_HCA` (e.g. `mlx5_1:1`),
`FABRIC_IFACES` (comma list), `FABRIC_IFACE0`, `FABRIC_SUBNET` (CIDR). Optional: `DSV41_HOME` (default `$HOME/dsv41`),
`SERVED_NAMES` (default `deepseek-v41-flash`). Images are rebuilt per node from identical inputs, so image IDs differ
across ranks; the launcher does not compare them - compare the build-input tarball md5 if you need identity. A
`GATE_ONLY=1` dry run of the launcher is safe on a serving rank: the gate runs before any container is removed.

**Eager-mode levers, measured (2026-09-11, same 1/3/6-stream battery; production = 32.4 / 58.4 / 95.4 tok/s aggregate):**
`--async-scheduling` is unstable with DSpark here — 3-stream runs alternated 46.3 and 15.9 tok/s, the slow runs being every
request taking ~37 s (a scheduling stall), single-stream 29.7; rejected. `torch.compile` without CUDA graphs
(`cudagraph_mode=NONE`) is correct (gate PASS, 7/7, no Xids) and within noise of plain eager: 28.8 / 60.4 / 93.3; rejected.
DSpark acceptance is 55% (about 2.75 of 5 drafts per step), so raising `k` is not a lever either. Single-stream throughput on
this tree is launch-bound at ~120 ms per decode step; only graph replay removes that, which is the open defect above.
Aggregate throughput scales with concurrency (about 60 at 3 streams, 90-95 at 6).

### Gotchas that cost node reboots

* **CRLF.** A Windows git checkout with `core.autocrlf=true` turns the reference's `mounts.txt` and patch files CRLF. A bind-mount
  target with a trailing CR is silently created as a *new file Python never imports*, so every patch is inactive while
  looking mounted. With the engram-on-disk patch inactive each rank loads ~47 GiB of Engram tables on top of ~79 GiB of
  backbone -> deterministic `NVRM: NV_ERR_NO_MEMORY` on every rank at the same second -> the hang-guard sysctls panic the node.
  Three fleet-wide reboots before a 60-second no-GPU import test found it. Checking that files matched *each other* across
  nodes proved nothing; check them against the **reference's** md5 table, and scan for CR bytes.
* **Memory margins were a red herring.** `gpu_memory_utilization`, `vm.min_free_kbytes`, and a page-cache-drop loop changed
  nothing — the overload was 47 GiB of tables that should not have been in memory. Docker `--memory` does **not** contain
  NVRM allocations either.
* **The post-load dip is normal.** After the last shard, MemAvailable falls from ~22 to ~7 GiB during
  `process_weights_after_loading` and recovers before KV allocation. A kill-switch above ~10 GiB fires spuriously there.
* **Preserve the engine log on the host** before any `docker rm -f`; a later preflight's `rm -f` destroyed the only log of a
  failed boot.

## How this differs from tonyd2wild's recipe (and why)

| aspect | tonyd2wild | here | why |
|---|---|---|---|
| base image | `vllm/vllm-openai:nightly-8a728663…` (the `dsv41-feat` merge-base) + the branch python tree copied in (`overlay1`) | the official `vllm/vllm-openai:deepseekv41-flash-0909-arm64` tag + the branch python tree copied over it (`Dockerfile.branch`) | the 0909 tag already ships a `_C_stable_libtorch` that loads on GB10 and a `deepseek_v4_1` tree; only its `engram.py` revision is incompatible with the branch patches, so the tree is swapped |
| compiled extension | `_C_stable_libtorch` rebuilt (`build_stable_ext.sh`, cutlass, -j20) | first the 0909 image's own build; then **also rebuilt** from the branch csrc (6 min) as `vllm-dsv41:branch-ext` | the rebuild is cheap and correct (`sm_120`/`sm_120f`; 12.1a maps to the 12.0 family in this tree) — but it did **not** fix graph mode, so it is not what makes their graphs boots work |
| FlashInfer | 0.7.0rc1 from source, pinned submodules (`overlay3`) + kernel prewarms (`overlay4/5`) | same (`Dockerfile.fi07`, `Dockerfile.v41`) | 0.6.18 lacks the SM120 sparse-MLA decode for top-k 1152 |
| patches | the 7 files bind-mounted from `~/patches/dsv41-boot10` | identical files, identical mounts | byte-for-byte; **verify md5 against their `patch/README.md`** (CRLF gotcha) |
| checkpoint placement | head on NVMe, workers over NFS; per-rank Engram rows copied locally (`engram_local.py`) | the full checkpoint local on **every** rank; Engram read directly from the local shards | no NFS, no `ENGRAM_LOCAL` step; loads 48 shards in ~45 s |
| serving config | boot 10: CUDA graphs `FULL_AND_PIECEWISE` + DSpark k=5, 300K, gmu 0.80, vision + tools — **~74 tok/s** | **eager + DSpark k=5**, 131K, gmu 0.80, vision + tools — **~30-35 tok/s** | graphs unusable on the unrebuilt extension; DSpark alone gives 2.2x |
| launcher | `dsv41-tp4.sh` + `prelaunch-*.sh` md5 checks, `postcheck10.sh`, `dgx-anti-oom` | `launch41.sh` with the gate **built in** (md5 vs their table, CR scan, in-container import/marker proof), host-side log capture, a `docker kill` kill-switch at MemAvailable < 2 GiB | three node reboots taught that the gate must be fail-closed and inside the launcher; `dgx-anti-oom` is not on this fleet |
| CUDA-graph bisect on this image | n/a (their graphs boots work) | **graphs ON, DSpark OFF → HTTP 200 with garbled/empty output**, on both the 0909 extension and the rebuilt one; graphs ON + DSpark → illegal memory access in the MoE reduce | corruption is in graph replay and is NOT the extension; DSpark merely trips over it. Never serve `EAGER=0` on this image lineage |

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

### `hash_moe` is not engram, and V4.1 does not use it at all

An earlier version of these notes said engram should attach at V4's existing `hash_moe` /
`num_hash_layers` plumbing. **That is wrong.** `hash_moe` is hashed expert *routing* - a
token-id -> expert-id table at `layers.N.ffn.gate.tid2eid`, gated by a prefix rule
(`layer_index < num_hash_layers`). The checkpoint settles what V4.1 does with it:

```
tid2eid/tie2eid tensors in V4.1 : 0
engram layer indices            : [1, 14]
layers with a normal ffn.gate.* : 40
```

Zero. V4.1 **replaced** hashed routing with engram rather than extending it. A V4.1 config
must therefore set `num_hash_layers = 0` to disable that path. Setting it to
`len(engram_layer_ids)` would route two layers through the wrong MoE implementation.

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

  **This is now measured both ways.** Uncapped, the same workload rebooted a node and the
  global OOM-killer took an unrelated user service with it. Capped at 0.82, an
  over-allocation on the same hardware produced a contained `SIGKILL` of the offending
  process only: no reboot, no kernel OOM-kill event on that node, co-user services still
  `active`. The cap converts a cluster-wide incident into a process-level failure.

### Loading is tighter than converting — expect to tune this (hypothesis, not both-arms tested)

With engram on disk the resident backbone is 74.5 GiB of a ~121 GiB unified pool. Weight
loading then has to read 74.5 GiB of safetensors **through page cache that shares that same
pool**, so the headroom during load is much thinner than the steady-state figure suggests.

On a 4-node run the node with the least free memory (115 GiB available vs 117 on its peers,
because of a couple of unrelated resident containers) OOMed during load while the others were
fine. `torchrun` then `SIGKILL`ed every other rank, so **all four logs show `exitcode -9` and
none of them identifies the culprit** — find it by checking which node actually logged kernel
OOM events (`journalctl -b 0 | grep -ci oom`), not by reading torchrun's timestamps, which
record when each agent noticed rather than when anything failed.

Practical consequences: size for your *tightest* node, not your average one; stop unrelated
containers on the ranks first; and consider a lower cap with correspondingly lower
`max_seq_len` / `max_batch_size` while bringing a new deployment up.

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

## What the V4-only build was missing (historical — superseded by the Served section)

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

There is a **fifth** delta, and it is the one that actually blocks serving. Found by running
it, not by reading it:

### 5. vLLM's compressed-attention is hardcoded to V4's pooling ratios

`compress_ratios` is a per-layer list in both models, but the values differ in kind:

```
V4-Flash : [0, 0, 4, 128, 4, 128, ..., 4, 0, 0, 0]   distinct {0, 4, 128}  interleaved
V4.1     : [0, 0, 2x18,  1x20,        0, 0, 0]       distinct {0, 1, 2}    blocked
```

They are literal pooling ratios in both cases - the reference is explicit: *"Pools
`compress_ratio` consecutive tokens into one KV latent"*, and it divides `max_seq_len` by the
value. V4.1 pools 2 tokens where V4 pooled 4 or 128, taking its savings from CED and
KV-sharing instead. V4.1's blocked structure mirrors its architecture: layers 0-19 encoder,
20-39 decoder, then 3 MTP.

vLLM cannot express that. Booting V4.1 dies on:

```
File ".../vllm/models/deepseek_v4/compressor.py", line 152, in __init__
    assert compress_ratio in [4, 128]
AssertionError
```

and it is **not just the assert**. Six places branch on the literal values:

```python
coff = 1 + (compress_ratio == 4)
self.sliding_window = coff * compress_ratio
if   compress_ratio == 4:   self.block_size = 4
elif compress_ratio == 128: self.block_size = 8
else: raise ValueError(f"Invalid compress ratio: {compress_ratio}")
self.overlap = compress_ratio == 4
... and self.compress_ratio == 128
```

`block_size` is not a free parameter. From vLLM's own comment:

> Block size is constrained by tensor sharing between compressor states and KV blocks. Since
> compressor states share the same physical tensor as KV blocks, they must use the same page
> size. **TODO(yifan): make block size automatically determined and configurable.**

Supporting ratios 1 and 2 means deriving the page-size arithmetic that currently yields 4 and
8, plus the right `coff` / `sliding_window` / `overlap`, plus whatever
`compress_norm_rope_store_triton` assumes. Guessing does not raise - it mismatches pages.
**Retracted 2026-09-11:** this was true of the **V4-only** vLLM build inspected (`deepseek_v4` package). The official
`vllm/vllm-openai:deepseekv41-flash-*` images and vLLM branch `dsv41-feat` ship a separate `deepseek_v4_1` package whose
compressor handles ratios 1 and 2. None of the analysis below is needed to serve V4.1; it is kept as a record of the V4 build.

---

So: five deltas. Engram needs new machinery but has a clean seam. CED, the candidate pool and
the DSpark MoE are wiring. On a **V4-only** build the compressor would need that work. On a `deepseek_v4_1` build (`dsv41-feat`), `vllm serve` loads and
serves V4.1 — see the Served section. The five-delta list below describes the V4 build only.

1. **Engram** — 6 tensor shapes across 2 layers (`embed.weight/.scale`, `wkv.weight/.scale`,
   `q_weight`, `k_weight`).

   The reference applies it to the hyper-connection residual stream **between the previous
   layer's `post` and this layer's `pre`** — and vLLM fuses exactly those two into
   `mhc_fused_post_pre_tilelang`, so there is normally no seam there. Two primitives vLLM
   already has create one without patching any vLLM source:

   * `mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)` reconstructs precisely
     the reference's `h` (vLLM already calls it to build draft-model aux states);
   * the decoder layer's `residual is None` branch runs an **unfused** `mhc_pre` on whatever
     `x` it is handed.

   So: reconstruct `h`, apply engram, hand it back as `x` with `residual=None`. That
   reproduces `post -> engram -> pre` exactly, at the cost of losing post/pre fusion on 2 of
   40 layers. [`tools/vllm_dsv41.py`](tools/vllm_dsv41.py) implements this as a layer
   wrapper. **It is written, reviewed and unproven — not yet run end-to-end.**

   The genuinely hard part is not the module, it is the n-gram history: hashing needs each
   token's predecessors, and the reference relies on a static right-padded batch with an
   absolute-position cache. Under continuous batching that assumption is gone. The draft
   yields the pad id wherever history is unavailable (the same thing the reference does at a
   sequence start), which is correct at a boundary and conservative mid-sequence — and it
   degrades quality **silently** when wrong. Diff logits against the reference on identical
   prompts before trusting it; there is a helper for that at the bottom of the module.
2. **CED** — decoder layers project global KV from the final encoder hidden states rather than
   their own, driven by `kv_source_layer_ids` / `index_source_layer_ids`. No new weights, and
   **vLLM already has the primitive**: `kv_sharing_target_layer_name` is a first-class
   cross-layer KV sharing mechanism (~50 files reference it; `v1/worker/gpu/attn_utils.py`
   skips cache allocation for layers that share a target, and Gemma-4's speculator uses it).
   Mapping `kv_source_layer_ids` onto it looks like wiring rather than invention — which also
   means the KV-cache accounting comes for free.
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
