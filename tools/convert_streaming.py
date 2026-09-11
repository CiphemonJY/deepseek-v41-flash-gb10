#!/usr/bin/env python3
"""
Per-rank streaming converter for DeepSeek-V4.1-Flash. Drop-in replacement for the
reference inference/convert.py on a 128 GB node.

WHY THIS EXISTS
  The reference convert.py builds `state_dicts = [{} for _ in range(mp)]` and fills ALL
  mp shards while walking the inputs, then calls save_file() at the very end. Peak host
  RAM is therefore the whole checkpoint -- 475 GiB. A DGX Spark has 121 GiB. It OOMs.

WHAT CHANGED (three things, nothing else)
  1. RANK IS THE OUTER LOOP. We re-read the inputs once per rank and hold only that
     rank's share: 286 GiB backbone / mp. At mp=4 that is ~71.5 GiB, which fits.
     Cost: mp passes over 475 GiB of NVMe instead of one.
  2. ENGRAM IS SPLIT OUT to engram{i}-mp{mp}.safetensors, so the backbone file stays
     ~71.5 GiB and the 189 GiB of n-gram tables can be memory-mapped at serve time
     instead of resident. On 4 nodes this is REQUIRED, not an optimisation --
     475 GiB / 4 = 118.8 GiB/node exceeds the 117 GiB a node actually has.
  3. ENGRAM USES get_slice(), NOT get_tensor(). layers.N.engram.embed.weight is a
     single ~91.5 GiB tensor; get_tensor() would load all of it just to keep 1/mp.
     get_slice()[a:b] reads only this rank's rows (~23 GiB at mp=4).

UNCHANGED from the reference: the name mapping, expert sharding, the wo_a block-scale
dequant to bf16, cast_e2m1fn_to_e4m3fn for --expert-dtype fp8, the fp4 view for fp4,
the ceil+pad row sharding of engram, and the tokenizer copy. Output for the backbone is
byte-identical to the reference given the same inputs.

USAGE
  python3 convert_streaming.py --hf-ckpt-path /path/to/hf --save-path /path/to/TP4 \
      --model-parallel 4 --expert-dtype fp4 --tokenizer-path /path/to/hf
  # one rank at a time (resumable, or spread across machines):
  python3 convert_streaming.py ... --ranks 0,1
"""
import json
import os
import re
import shutil
from argparse import ArgumentParser
from glob import glob

import torch
from safetensors.torch import safe_open, save_file
from tqdm import tqdm

FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

ENGRAM_MARKER = ".engram.embed."


def cast_e2m1fn_to_e4m3fn(x: torch.Tensor, scale: torch.Tensor):
    """Casts a tensor from e2m1fn to e4m3fn losslessly. Verbatim from the reference."""
    assert x.dtype == torch.int8
    assert x.ndim == 2
    out_dim, in_dim = x.size()
    in_dim *= 2
    fp8_block_size = 32
    fp4_block_size = 32
    assert in_dim % fp8_block_size == 0 and out_dim % fp8_block_size == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_block_size

    x = x.view(torch.uint8)
    low = x & 0x0F
    high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)

    MAX_OFFSET_BITS = 6  # 6.0 * 2^6 = 384 < 448 (e4m3fn max)

    bOut = out_dim // fp8_block_size
    bIn = in_dim // fp8_block_size
    x = x.view(bOut, fp8_block_size, bIn, fp8_block_size).transpose(1, 2)
    scale = scale.float().view(bOut, fp8_block_size, bIn, -1).transpose(1, 2).flatten(2)
    scale_max_offset_bits = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
    offset = scale / scale_max_offset_bits
    offset = offset.unflatten(-1, (fp8_block_size, -1)).repeat_interleave(fp4_block_size, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn), scale_max_offset_bits.squeeze(-1).to(torch.float8_e8m0fnu)


mapping = {
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}


def infer_num_experts(names):
    """Number of routed experts in the backbone and in the MTP layers, from the weight names."""
    counts = [0, 0]
    for name in names:
        name = name.removeprefix("model.")
        match = re.search(r"(?:mlp|ffn)\.experts\.(\d+)\.", name)
        if match:
            is_mtp = name.startswith("mtp.")
            counts[is_mtp] = max(counts[is_mtp], int(match.group(1)) + 1)
    assert counts[0], "no routed experts found in the checkpoint"
    return counts[0], counts[1] or counts[0]


def canonical_name(source_name):
    """The reference's rename chain. Returns (new_name, shard_dim_or_None, key)."""
    name = source_name
    if name.startswith("model."):
        name = name[len("model.") :]
    if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
        return None, None, None  # MTP ties these to the backbone
    name = name.replace("self_attn", "attn")
    if not name.startswith("vision."):
        name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")
    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "tid2eid", "ape", "image_"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]
    if key in mapping:
        new_key, dim = mapping[key]
    else:
        new_key, dim = key, None
    return name.replace(key, new_key), dim, key


def postprocess(sd, expert_dtype):
    """wo_a block-scale dequant + expert dtype handling. Verbatim semantics, in-place."""
    for name in list(sd.keys()):
        if name.endswith("wo_a.weight"):
            weight = sd[name]
            scale = sd.pop(name.replace("weight", "scale"))
            assert weight.size(0) % scale.size(0) == 0
            assert weight.size(1) % scale.size(1) == 0
            out_block_size = weight.size(0) // scale.size(0)
            in_block_size = weight.size(1) // scale.size(1)
            assert (out_block_size, in_block_size) in ((32, 32), (128, 128)), (
                name, weight.shape, scale.shape,
            )
            weight = (
                weight.unflatten(0, (-1, out_block_size)).unflatten(-1, (-1, in_block_size)).float()
                * scale[:, None, :, None].float()
            )
            sd[name] = weight.flatten(2, 3).flatten(0, 1).bfloat16()
        elif "experts" in name and sd[name].dtype == torch.int8:
            if expert_dtype == "fp8":
                scale_name = name.replace("weight", "scale")
                weight = sd.pop(name)
                scale = sd.pop(scale_name)
                sd[name], sd[scale_name] = cast_e2m1fn_to_e4m3fn(weight, scale)
            else:
                sd[name] = sd[name].view(torch.float4_e2m1fn_x2)


def convert_rank(rank, files, mp, n_experts, mtp_n_experts, expert_dtype, save_path):
    """One pass over every input file, keeping only this rank's share."""
    backbone, engram = {}, {}

    for file_path in tqdm(files, desc=f"rank {rank}/{mp}", unit="shard"):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for source_name in f.keys():
                name, dim, _ = canonical_name(source_name)
                if name is None:
                    continue

                # --- engram: slice rows lazily, never materialise the full table ---
                if ENGRAM_MARKER in name:
                    sl = f.get_slice(source_name)
                    rows = sl.get_shape()[0]
                    shard_size = (rows + mp - 1) // mp   # ceil, as the reference does
                    lo = rank * shard_size
                    hi = min(lo + shard_size, rows)
                    part = sl[lo:hi] if lo < rows else None
                    if part is None:
                        part = torch.empty((0, sl.get_shape()[1]), dtype=sl.get_dtype()
                                           if hasattr(sl, "get_dtype") else torch.uint8)
                    if part.size(0) < shard_size:        # pad the tail shard
                        pad_value = 1 if name.endswith(".scale") else 0
                        padding = part.new_full((shard_size - part.size(0), part.size(1)), pad_value)
                        part = torch.cat([part, padding])
                    engram[name] = part.contiguous()
                    continue

                # --- experts: this rank owns a contiguous block of expert indices ---
                if "experts" in name and "shared_experts" not in name:
                    current = mtp_n_experts if name.startswith("mtp.") else n_experts
                    n_local = current // mp
                    idx = int(name.split(".")[-3])
                    if idx < rank * n_local or idx >= (rank + 1) * n_local:
                        continue
                    backbone[name] = f.get_tensor(source_name)
                    continue

                # --- everything else: narrow along the mapped dim, or replicate ---
                if dim is not None:
                    param = f.get_tensor(source_name)
                    assert param.size(dim) % mp == 0, f"Dimension {dim} must be divisible by {mp}"
                    shard = param.size(dim) // mp
                    backbone[name] = param.narrow(dim, rank * shard, shard).contiguous()
                    del param
                else:
                    backbone[name] = f.get_tensor(source_name)

    postprocess(backbone, expert_dtype)

    out_main = os.path.join(save_path, f"model{rank}-mp{mp}.safetensors")
    save_file(backbone, out_main)
    backbone.clear()

    out_eng = os.path.join(save_path, f"engram{rank}-mp{mp}.safetensors")
    save_file(engram, out_eng)
    engram.clear()

    for p in (out_main, out_eng):
        print(f"  wrote {p}  ({os.path.getsize(p) / 2**30:.1f} GiB)")


def main(hf_ckpt_path, save_path, mp, expert_dtype, tokenizer_path=None, ranks=None):
    torch.set_num_threads(8)
    os.makedirs(save_path, exist_ok=True)

    index_path = os.path.join(hf_ckpt_path, "model.safetensors.index.json")
    expected = set(json.load(open(index_path))["weight_map"]) if os.path.exists(index_path) else None

    files = sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))
    assert files, f"no safetensors under {hf_ckpt_path}"

    all_names = expected
    if all_names is None:
        all_names = set()
        for fp in files:
            with safe_open(fp, framework="pt", device="cpu") as f:
                all_names.update(f.keys())
    else:
        # fail fast on a partial download rather than 4 passes in
        seen = set()
        for fp in files:
            with safe_open(fp, framework="pt", device="cpu") as f:
                seen.update(f.keys())
        assert seen == expected, (
            f"checkpoint incomplete: {len(expected - seen)} tensors missing, "
            f"{len(seen - expected)} unexpected (source may be mid-upload)"
        )

    n_experts, mtp_n_experts = infer_num_experts(all_names)
    assert n_experts % mp == 0 and mtp_n_experts % mp == 0, (n_experts, mtp_n_experts, mp)
    print(f"{n_experts=} {mtp_n_experts=} {mp=} expert_dtype={expert_dtype}")

    todo = ranks if ranks is not None else list(range(mp))
    for rank in todo:
        assert 0 <= rank < mp, rank
        convert_rank(rank, files, mp, n_experts, mtp_n_experts, expert_dtype, save_path)

    tokenizer_path = tokenizer_path or hf_ckpt_path
    for fn in ["tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(tokenizer_path, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(save_path, fn))


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--hf-ckpt-path", type=str, required=True)
    p.add_argument("--save-path", type=str, required=True)
    p.add_argument("--model-parallel", type=int, required=True)
    p.add_argument("--expert-dtype", type=str, choices=["fp8", "fp4"], default=None)
    p.add_argument("--tokenizer-path", type=str, default=None)
    p.add_argument("--ranks", type=str, default=None,
                   help="comma-separated subset, e.g. 0,1 -- for resuming or splitting across boxes")
    a = p.parse_args()
    rk = [int(x) for x in a.ranks.split(",")] if a.ranks else None
    main(a.hf_ckpt_path, a.save_path, a.model_parallel, a.expert_dtype, a.tokenizer_path, rk)
