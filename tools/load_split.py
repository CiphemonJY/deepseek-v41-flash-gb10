"""Load a split DeepSeek-V4.1-Flash checkpoint: resident backbone + mmap-backed engram.

convert_streaming.py writes two files per rank:
    model{rank}-mp{mp}.safetensors    74.5 GiB  -- backbone, loaded resident
    engram{rank}-mp{mp}.safetensors   47.2 GiB  -- n-gram tables, left on NVMe

The reference generate.py calls safetensors `load_model()` on the single model file, which
rejects a split checkpoint with missing keys. This module replaces that one call.

WHY ENGRAM STAYS ON DISK
    74.5 + 47.2 = 121.7 GiB per rank against 117 GiB available on a 128 GB DGX Spark.
    Resident engram does not fit. It does not need to: the tables are pure lookup, and
    a decode step touches (max_ngram_size-1) * n_heads = 24 rows per engram layer --
    about 12 KB of the 47 GiB. `safe_open(...).get_tensor()` returns an mmap-backed CPU
    tensor at zero resident cost (measured: 0.49 GiB RSS holding a 22.9 GiB fp8 table),
    and F.embedding gathers straight out of it, paging in only the rows touched.

MEASURED on a node, GB10 / SM 12.1, torch 2.13.0+cu130:
    get_tensor(22.9 GiB fp8 weight)  -> RSS unchanged at 0.49 GiB   (mmap confirmed)
    F.embedding on an fp8 CPU mmap   -> OK
    dequant * e8m0 scale, block=32   -> bfloat16, finite

The mmap handles are held in _HANDLES for process lifetime. Dropping them unmaps the
tables and every subsequent lookup segfaults, so do not clear that list.
"""
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_model

_HANDLES = []  # safe_open handles; must outlive every lookup

# How much checkpoint to read before releasing the mmap and reopening it. Bounds host RSS
# growth during load; tune down if a node is tighter, up to trade memory for fewer reopens.
_RELEASE_EVERY_BYTES = int(os.environ.get("DSV41_LOAD_WINDOW_GIB", "6")) * 2**30


def guard_cuda_memory(fraction: float = 0.82) -> None:
    """Cap this process's share of the unified pool so an overrun raises a Python OOM.

    INCIDENT 2026-09-11: without this, a 121.7 GiB allocation against a 121.7 GiB unified
    pool did not fail politely -- it starved the kernel, the global OOM-killer reaped an
    unrelated service (an unrelated user service), and a node hard-rebooted. On GB10 the GPU
    and host share one LPDDR5X pool, so a CUDA overcommit is a SYSTEM overcommit. Always
    cap before building a model whose size you are still estimating.
    """
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"[load_split] CUDA capped at {fraction:.0%} of {total:.1f} GiB "
              f"= {total * fraction:.1f} GiB (overrun -> Python OOM, not a box reboot)")


def patch_engram_for_mmap(model_module) -> None:
    """Stop engram tables being allocated at all. MUST run BEFORE Transformer(...).

    ParallelEngramEmbedding.__init__ creates self.weight / self.scale as real nn.Parameters.
    generate.py builds the model inside `with torch.device("cuda")`, so those 47.2 GiB of
    tables land on the GPU at CONSTRUCTION time -- long before any loader could swap them
    for mmap views. Backbone 74.5 + engram 47.2 = 121.7 GiB against a 121.7 GiB pool.

    Building them on the `meta` device costs zero bytes while keeping them in state_dict(),
    so load_model(strict=False) still reports them missing and load_split() can verify the
    split is exactly as expected before replacing them with mmap tensors.
    """
    cls = model_module.ParallelEngramEmbedding
    if getattr(cls, "_mmap_patched", False):
        return
    orig_init = cls.__init__

    def meta_init(self, num_embeddings, dim):
        with torch.device("meta"):
            orig_init(self, num_embeddings, dim)

    cls.__init__ = meta_init
    cls.forward = _mmap_engram_forward
    cls._mmap_patched = True
    print("[load_split] engram tables will be built on `meta` (0 bytes) and mmap-backed")


def _mmap_engram_forward(self, indices: torch.Tensor) -> torch.Tensor:
    """ParallelEngramEmbedding.forward against a CPU mmap instead of a CUDA Parameter.

    Identical arithmetic to the reference: mask foreign rows, gather, dequantize per
    32-wide block, zero the masked positions, then all_reduce so every rank ends up with
    the union. Only the device dance is new -- the gather runs on CPU because that is
    where the mmap lives, and GB10's unified memory makes the 12 KB hop ~free.
    """
    device = indices.device
    mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
    local = (indices - self.vocab_start_idx).masked_fill(mask, 0)

    local_cpu = local.to("cpu", non_blocking=False)
    values = F.embedding(local_cpu, self.weight)   # fp8 gather out of the mmap
    scales = F.embedding(local_cpu, self.scale)
    values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
    values = values.flatten(-2).to(torch.bfloat16).to(device)

    values = values.masked_fill(mask.unsqueeze(-1), 0)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(values)
    return values


def _stream_load_backbone(model, path: str, verbose: bool):
    """Copy each tensor from the checkpoint into the model, one at a time.

    Returns (missing, unexpected) with the same meaning as safetensors' load_model(...,
    strict=False), so the engram verification downstream is unchanged.

    Parameters still on `meta` (the engram tables) are never touched: their keys do not
    appear in the backbone file, so they fall through to `missing` -- which is exactly what
    load_split() asserts on.
    """
    state = model.state_dict()
    unexpected: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = list(f.keys())

    # Per-tensor copying alone is NOT enough. `del src` drops the Python reference, but the
    # mmap's file-backed pages stay mapped, so RSS climbs monotonically as we walk the file:
    # measured 30.5 GiB of host RSS at 40% through a 74.5 GiB checkpoint, which drove the box
    # into swap (7.1 GiB) with only 10 GiB available. Closing and reopening the handle
    # munmaps and releases those pages, bounding RSS to roughly one window.
    done = 0
    copied_bytes = 0
    i = 0
    while i < len(keys):
        with safe_open(path, framework="pt", device="cpu") as f:
            window_bytes = 0
            while i < len(keys) and window_bytes < _RELEASE_EVERY_BYTES:
                key = keys[i]
                i += 1
                target = state.get(key)
                if target is None:
                    unexpected.append(key)
                    continue
                src = f.get_tensor(key)
                if tuple(target.shape) != tuple(src.shape):
                    raise RuntimeError(
                        f"{key}: checkpoint {tuple(src.shape)} != model {tuple(target.shape)}")
                nbytes = src.numel() * src.element_size()
                with torch.no_grad():
                    target.copy_(src)      # H2D; casts if dtypes differ, as load_model does
                del src
                window_bytes += nbytes
                copied_bytes += nbytes
                done += 1
        # handle closed here -> mapping released before the next window opens
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if verbose:
            print(f"[load_split] {done}/{len(keys)} tensors, "
                  f"{copied_bytes / 2**30:.1f} GiB copied", flush=True)
    # Anything the model wants that the backbone file did not provide -- EXCEPT tied
    # weights. The MTP layers tie their token embedding and output head to the backbone's,
    # so convert.py deliberately omits `mtp.N.embed.weight` / `mtp.N.head.weight`:
    #
    #     if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
    #         continue   # an MTP layer ties its ... to the backbone's
    #
    # safetensors' load_model() handles this by detecting tensors that share storage and
    # requiring only one copy in the file. A hand-rolled streaming loader has to do the same
    # or it reports six phantom missing keys. A tied key needs no copy: writing the backbone's
    # tensor already wrote it.
    present = set(keys)
    loaded_ptrs: dict[int, str] = {}
    for k in present:
        t = state.get(k)
        if t is not None and t.device.type != "meta":
            loaded_ptrs[t.data_ptr()] = k

    missing, tied = [], {}
    for k, t in state.items():
        if k in present:
            continue
        if t.device.type != "meta" and t.data_ptr() in loaded_ptrs:
            tied[k] = loaded_ptrs[t.data_ptr()]   # shares storage with something we wrote
        else:
            missing.append(k)
    if verbose and tied:
        shown = ", ".join(f"{k} <- {v}" for k, v in list(tied.items())[:3])
        print(f"[load_split] {len(tied)} tied weight(s) satisfied by shared storage ({shown}"
              f"{', ...' if len(tied) > 3 else ''})", flush=True)
    if verbose:
        print(f"[load_split] streamed {len(keys) - len(unexpected)} tensors, "
              f"{len(missing)} left for the engram file")
    return missing, unexpected


def load_split(model, ckpt_path: str, rank: int, world_size: int, verbose: bool = True):
    """Drop-in for `load_model(model, f"{ckpt_path}/model{rank}-mp{world_size}.safetensors")`."""
    main_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    eng_path = os.path.join(ckpt_path, f"engram{rank}-mp{world_size}.safetensors")
    for p in (main_path, eng_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # 1. backbone, resident, STREAMED one tensor at a time.
    #
    # safetensors' load_model() takes a bulk path: the mmap'd CPU side and the 74.5 GiB of
    # device parameters are both live at peak. On a unified-memory box those share ONE pool,
    # so peak approaches 2x the checkpoint and the tightest node OOMs during load even though
    # steady state fits comfortably. Measured: a 4-rank load died on the node with 115 GiB
    # available while its 117 GiB peers survived.
    #
    # Copying per tensor and dropping each source immediately keeps host residency flat --
    # only one tensor is live at a time, and the page cache behind it becomes reclaimable
    # straight away.
    missing, unexpected = _stream_load_backbone(model, main_path, verbose and rank == 0)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {main_path}: {unexpected[:5]} ...")

    # 2. engram, mmap. Every missing key must be an engram key and vice versa --
    #    otherwise something else failed to load and we would serve garbage silently.
    handle = safe_open(eng_path, framework="pt", device="cpu")
    _HANDLES.append(handle)
    eng_keys = set(handle.keys())

    missing = set(missing)
    if missing != eng_keys:
        raise RuntimeError(
            "split checkpoint mismatch.\n"
            f"  backbone missing but not in engram file: {sorted(missing - eng_keys)[:8]}\n"
            f"  in engram file but backbone did not want: {sorted(eng_keys - missing)[:8]}"
        )

    # 3. swap each ParallelEngramEmbedding's Parameters for the mmap tensors, and its
    #    forward for the CPU-gather version.
    patched = 0
    by_name = dict(model.named_modules())
    for key in sorted(eng_keys):
        mod_name, attr = key.rsplit(".", 1)        # layers.14.engram.embed . weight
        mod = by_name.get(mod_name)
        if mod is None:
            raise RuntimeError(f"no module {mod_name} for engram key {key}")
        tensor = handle.get_tensor(key)            # zero-copy mmap view
        expected = (mod.part_num_embeddings, mod.dim if attr == "weight" else mod.dim // mod.block_size)
        if tuple(tensor.shape) != expected:
            raise RuntimeError(f"{key}: shape {tuple(tensor.shape)} != expected {expected}")
        if attr in mod._parameters:
            del mod._parameters[attr]
        setattr(mod, attr, tensor)
        patched += 1

    for mod_name in {k.rsplit(".", 2)[0] + ".embed" for k in eng_keys}:
        mod = by_name[mod_name]
        mod.forward = _mmap_engram_forward.__get__(mod, type(mod))

    if verbose and rank == 0:
        n_tables = len({k.rsplit(".", 1)[0] for k in eng_keys})
        gib = os.path.getsize(eng_path) / 2**30
        print(f"[load_split] backbone resident from {os.path.basename(main_path)}")
        print(f"[load_split] engram mmap: {n_tables} tables, {patched} tensors, "
              f"{gib:.1f} GiB left on NVMe")
    return missing, unexpected
