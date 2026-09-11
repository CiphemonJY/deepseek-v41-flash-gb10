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


def load_split(model, ckpt_path: str, rank: int, world_size: int, verbose: bool = True):
    """Drop-in for `load_model(model, f"{ckpt_path}/model{rank}-mp{world_size}.safetensors")`."""
    main_path = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    eng_path = os.path.join(ckpt_path, f"engram{rank}-mp{world_size}.safetensors")
    for p in (main_path, eng_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # 1. backbone, resident. strict=False because engram lives in the other file.
    missing, unexpected = load_model(model, main_path, strict=False)
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
