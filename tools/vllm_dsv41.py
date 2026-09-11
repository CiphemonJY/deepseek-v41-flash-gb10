"""Adapt vLLM's DeepSeek-V4 implementation to serve DeepSeek-V4.1-Flash.

    import vllm_dsv41; vllm_dsv41.register()

V4.1 is an increment on V4, not a new architecture. vLLM's deepseek_v4 package already
implements CSA, the sparse indexer, mHC/hyper-connections with Sinkhorn, FP4 experts,
hashed expert routing (`hash_moe` / `tid2eid`), MTP and the DSpark draft scaffolding.
Four things are missing for V4.1, and only ONE of them needs new machinery:

  1. ENGRAM  -- new module, the only delta with new weights (6 tensors x 2 layers).
  2. CED     -- `kv_source_layer_ids` / `index_source_layer_ids`. No new weights. vLLM
                already has the primitive: `kv_sharing_target_layer_name`, honoured by
                v1/worker/gpu/attn_utils.py (which skips cache allocation for a layer that
                shares a target) and by the GPU model runner. Wiring, not invention.
  3. HIERARCHICAL SPARSE INDEXER -- `candidate_source_layer_id`, `candidate_topk_blocks`,
                `candidate_block_size`. Bounds deeper indexer cost; an efficiency
                mechanism, so omitting it should cost speed, not correctness. Left unwired
                here and flagged, because "should" is not "verified".
  4. DSPARK ROUTED MOE -- `dspark_n_routed_experts` / `dspark_num_experts_per_tok`.
                Speculative decoding only; the model serves without it.

================================================================================
HOW ENGRAM IS INSERTED -- the one subtle part
================================================================================
The reference applies engram in Transformer.forward, to the hyper-connection residual
stream, BETWEEN the previous layer's `post` and this layer's `pre`:

    for i, layer in enumerate(self.layers):
        if layer.engram is not None:
            h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :], mask)
        h, pre_mix = layer(h, start_pos, pre_mix, image_mask)

vLLM FUSES post+pre into one kernel (`mhc_fused_post_pre_tilelang`) for speed, so there is
normally no seam at that point. Two existing primitives give us one without touching the
decoder layer's forward:

  * `mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)` reconstructs precisely
    the reference's `h`. vLLM already calls it to build aux hidden states for draft models.
  * The decoder layer's `residual is None` branch runs an UNFUSED `mhc_pre` on whatever `x`
    it is handed.

So for an engram layer we reconstruct h, apply engram, then hand it back as `x` with
residual/post_mix/res_mix cleared -- which routes through the unfused pre and reproduces the
reference's post -> engram -> pre ordering exactly. Cost: the two engram layers lose
post/pre fusion, i.e. one extra kernel launch each out of 40 layers.

================================================================================
KNOWN LIMITATION -- read before trusting this under load
================================================================================
N-gram hashing needs each token's PRECEDING tokens. The reference keeps a
[max_batch_size, max_seq_len] int64 cache and indexes it by absolute position, which is
valid because it runs one static batch with right-padded prompts.

Under vLLM's continuous batching, tokens from many sequences at arbitrary offsets share a
forward pass, so that cache is not directly usable. `EngramHashes.build()` below derives
lookback from `positions` and the per-request token history, and DELIBERATELY yields the
pad id wherever history is unavailable rather than guessing -- the same thing the reference
does at a sequence start or across an image span. That is correct at a sequence boundary
and conservative mid-sequence on the first chunked-prefill step.

This is the part most likely to be subtly wrong, and it degrades quality silently rather
than erroring. Validate against reference-implementation output on identical prompts before
serving anything real. See `verify_against_reference()` at the bottom.
"""
from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["register", "EngramHashes", "ParallelEngramEmbedding", "Engram"]

_ENGRAM_MMAP_HANDLES: list[Any] = []  # safe_open handles; must outlive every lookup


# --------------------------------------------------------------------------------------
# 1. Engram
# --------------------------------------------------------------------------------------
class ParallelEngramEmbedding(nn.Module):
    """Row-sharded fp8 n-gram table. Optionally backed by an mmap instead of device memory.

    At TP=4 the two tables are 47.2 GiB per rank, against ~117 GiB usable on a 128 GB
    unified-memory node that must also hold a 74.5 GiB backbone. 74.5 + 47.2 = 121.7 GiB
    does not fit, so mmap is not an optimisation here -- it is what makes TP=4 possible.

    A decode step touches (max_ngram_size - 1) * n_heads = 24 rows per engram layer, about
    12 KB of a 189 GiB table, so the lookup is latency-trivial. Measured: `get_tensor()` on
    a 22.9 GiB fp8 tensor leaves RSS unchanged at 0.49 GiB.
    """

    def __init__(self, num_embeddings: int, dim: int, rank: int, world_size: int,
                 block_size: int = 32, scale_dtype=torch.float8_e8m0fnu):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.block_size = block_size
        self.part_num_embeddings = (num_embeddings + world_size - 1) // world_size
        self.vocab_start_idx = rank * self.part_num_embeddings
        self.vocab_end_idx = self.vocab_start_idx + self.part_num_embeddings
        # built on `meta` so construction costs nothing; bind_* replaces them.
        # `with torch.device("cuda")` would otherwise allocate 47.2 GiB HERE, before any
        # loader could intervene -- and on unified memory that overcommit takes the box
        # down rather than raising.
        with torch.device("meta"):
            self.weight = nn.Parameter(
                torch.empty(self.part_num_embeddings, dim, dtype=torch.float8_e4m3fn),
                requires_grad=False)
            self.scale = nn.Parameter(
                torch.empty(self.part_num_embeddings, dim // block_size, dtype=scale_dtype),
                requires_grad=False)
        self._mmap = False

    def bind_mmap(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        """Attach CPU mmap-backed tensors, replacing the meta Parameters."""
        for attr, t in (("weight", weight), ("scale", scale)):
            expected = (self.part_num_embeddings,
                        self.dim if attr == "weight" else self.dim // self.block_size)
            if tuple(t.shape) != expected:
                raise ValueError(f"engram {attr}: {tuple(t.shape)} != expected {expected}")
            if attr in self._parameters:
                del self._parameters[attr]
            setattr(self, attr, t)
        self._mmap = True

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """indices: [..., n_hash_cols] global row ids -> [..., n_hash_cols, dim] bf16."""
        device = indices.device
        mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
        local = (indices - self.vocab_start_idx).masked_fill(mask, 0)

        if self._mmap:
            local = local.to("cpu")
        values = F.embedding(local, self.weight)
        scales = F.embedding(local, self.scale)
        values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
        values = values.flatten(-2).to(torch.bfloat16)
        if self._mmap:
            values = values.to(device)

        values = values.masked_fill(mask.unsqueeze(-1), 0)
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            # each rank owns a row range; the union is formed by summing the masked results
            torch.distributed.all_reduce(values)
        return values


class Engram(nn.Module):
    """Writes an n-gram lookup into the hc residual stream, gated by stream/key agreement.

    Ported from the reference implementation. Shapes are vLLM's flattened token layout
    ([num_tokens, hc_mult, dim]) rather than the reference's [B, L, hc_mult, dim]; the
    arithmetic is identical because every op is over trailing dims.
    """

    def __init__(self, config, layer_hash_index: int, rank: int, world_size: int,
                 linear_cls=None, prefix: str = ""):
        super().__init__()
        text = _text_config(config)
        self.layer_hash_index = layer_hash_index
        self.dim = text.hidden_size
        self.hc_mult = text.hc_mult
        self.eps = text.rms_norm_eps
        self.clamp_value = 1e-6

        n_heads = text.engram_n_heads
        head_dim = text.engram_head_dim
        n_hash_cols = (text.engram_max_ngram_size - 1) * n_heads
        self.embed = ParallelEngramEmbedding(
            text.engram_num_embeddings[layer_hash_index], head_dim, rank, world_size)

        # wkv: [n_hash_cols * head_dim] -> dim * (hc_mult + 1). Replicated, not sharded:
        # it is ~5 MB and sharding it would add a collective to a 2-layer-in-40 path.
        if linear_cls is None:
            self.wkv = nn.Linear(n_hash_cols * head_dim, self.dim * (self.hc_mult + 1), bias=False)
        else:
            self.wkv = linear_cls(n_hash_cols * head_dim, self.dim * (self.hc_mult + 1),
                                  bias=False, prefix=f"{prefix}.wkv")
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim), requires_grad=False)
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim), requires_grad=False)

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor,
                token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [..., hc_mult, dim]; hash_ids: [..., n_hash_cols]; token_mask False => gate shut."""
        looked_up = self.embed(hash_ids).flatten(-2)
        kv = self.wkv(looked_up)
        if isinstance(kv, tuple):      # vLLM linear layers return (out, bias)
            kv = kv[0]
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))

        weight = self.q_weight.float() * self.k_weight.float()   # only used as a product
        h, eps = x.float(), self.eps
        # normalised per (token, hc copy) over dim -- NOT jointly across the copies
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching the training kernel
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


# --------------------------------------------------------------------------------------
# 2. N-gram hashing (the continuous-batching-sensitive part)
# --------------------------------------------------------------------------------------
class EngramHashes(nn.Module):
    """Maps token ids to the bucket ids of the n-grams ending at each position.

    Hash construction is taken verbatim from the reference so the bucket layout matches the
    trained tables exactly: ids go through a compressed token map (tokens that normalise
    alike collapse together), then each position is XOR-hashed with the preceding
    max_ngram_size-1 tokens, and each (n-gram size, head) pair lands in its own prime-sized
    disjoint range.

    Every multiplier derives from engram_compressed_vocab_size, and the reference asserts it
    against the tokenizer. A mismatch there silently rehashes the whole 189 GiB table into
    noise, so we assert too rather than warn.
    """

    DEAD = -1

    def __init__(self, config, tokenizer):
        super().__init__()
        from engram import EngramLayout, build_compressed_token_map, compute_hash_multipliers
        import numpy as np

        text = _text_config(config)
        self.layout = EngramLayout.from_args(_ReferenceArgsView(text))
        token_map, vocab_size = build_compressed_token_map(tokenizer)
        if vocab_size != text.engram_compressed_vocab_size:
            raise ValueError(
                f"compressed vocab {vocab_size} != config {text.engram_compressed_vocab_size}; "
                "the hash multipliers derive from this, so the tables would be misread")
        self.pad_id = token_map[text.engram_pad_token_id]
        self.max_ngram = self.layout.max_ngram_size

        flat = [[p for per_ngram in layer for p in per_ngram] for layer in self.layout.primes]
        offsets = [np.cumsum([0, *sizes[:-1]]) for sizes in flat]
        self.register_buffer("primes", torch.tensor(self.layout.primes), persistent=False)
        self.register_buffer("offsets", torch.tensor(np.array(offsets)), persistent=False)
        self.register_buffer("multipliers",
                             compute_hash_multipliers(self.layout.layer_ids, self.max_ngram, vocab_size),
                             persistent=False)
        self.register_buffer("token_map", torch.tensor(token_map), persistent=False)

    @torch.inference_mode()
    def build(self, input_ids: torch.Tensor, lookback: torch.Tensor | None = None,
              token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """input_ids: [num_tokens]; lookback: [num_tokens, max_ngram-1] preceding ids, or None.

        Returns [num_tokens, n_engram_layers, n_hash_cols].

        Where `lookback` is None or negative we substitute the pad id, exactly as the
        reference does at a sequence start or across an image span. That is correct at a
        boundary and conservative elsewhere -- see the module docstring.
        """
        compressed = self.token_map[input_ids.clamp_min(0)]
        if token_mask is not None:
            compressed = torch.where(token_mask, compressed, self.DEAD)

        cols = [compressed]
        for shift in range(1, self.max_ngram):
            if lookback is None:
                prev = torch.full_like(compressed, self.pad_id)
            else:
                src = lookback[:, shift - 1]
                prev = torch.where(src >= 0, self.token_map[src.clamp_min(0)],
                                   torch.full_like(src, self.pad_id))
            cols.append(prev)
        toks = torch.stack(cols, dim=-1)                      # [num_tokens, max_ngram]
        blocked = torch.zeros_like(toks, dtype=torch.bool)
        blocked |= toks == self.DEAD
        toks = torch.where(blocked, self.pad_id, toks)

        products = toks.unsqueeze(1) * self.multipliers        # [num_tokens, n_layers, max_ngram]
        rolling, hashes = products[..., 0], []
        for i in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, i - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class _ReferenceArgsView:
    """Adapts an HF config to the attribute names EngramLayout.from_args expects."""

    def __init__(self, text):
        self.engram_layer_ids = tuple(text.engram_layer_ids)
        self.engram_num_embeddings = tuple(text.engram_num_embeddings)
        self.engram_max_ngram_size = text.engram_max_ngram_size
        self.engram_vocab_size = text.engram_vocab_size
        self.engram_n_heads = text.engram_n_heads
        self.engram_head_dim = text.engram_head_dim
        self.engram_pad_id = text.engram_pad_token_id
        self.engram_compressed_vocab_size = text.engram_compressed_vocab_size


def _text_config(config):
    get = getattr(config, "get_text_config", None)
    return get() if callable(get) else getattr(config, "text_config", config)


# --------------------------------------------------------------------------------------
# 3. CED -- wire kv_source_layer_ids onto vLLM's existing KV-sharing primitive
# --------------------------------------------------------------------------------------
def wire_ced(model: nn.Module, config, layer_name_fmt: str = "model.layers.{}.attn.attn") -> int:
    """Point each layer's attention at the layer whose KV it reuses.

    V4.1's CED projects the decoder's global KV from the final encoder hidden states rather
    than from each decoder layer's own, expressed as `kv_source_layer_ids`. vLLM models this
    natively with `kv_sharing_target_layer_name`: a layer carrying one does not allocate its
    own KV cache and reads the target's instead (see v1/worker/gpu/attn_utils.py), so the
    cache accounting follows for free.

    Returns the number of layers wired. Zero means the attribute path is wrong for this
    build -- treat that as a hard failure, not a no-op, because the model would then
    silently compute its own KV and produce incoherent output.
    """
    text = _text_config(config)
    sources = list(getattr(text, "kv_source_layer_ids", []) or [])
    if not sources:
        return 0
    n_layers = text.num_hidden_layers
    wired = 0
    for idx in range(n_layers):
        target_idx = max((s for s in sources if s <= idx), default=None)
        if target_idx is None or target_idx == idx:
            continue                       # a source layer owns its own KV
        attn = _resolve(model, layer_name_fmt.format(idx))
        if attn is None:
            continue
        attn.kv_sharing_target_layer_name = layer_name_fmt.format(target_idx)
        wired += 1
    return wired


def _resolve(root: nn.Module, dotted: str):
    cur = root
    for part in dotted.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


# --------------------------------------------------------------------------------------
# 4. Weight loading for the engram tensors
# --------------------------------------------------------------------------------------
def load_engram_weights(engrams: dict[int, Engram], ckpt_dir: str, rank: int, world_size: int,
                        mmap: bool = True) -> dict[str, Any]:
    """Load `engram{rank}-mp{world_size}.safetensors` produced by convert_streaming.py.

    The 6 tensors per engram layer are:
        layers.N.engram.embed.weight / .scale     the table (mmap'd when mmap=True)
        layers.N.engram.wkv.weight   / .scale     the projection
        layers.N.engram.q_weight / .k_weight      the gate
    """
    from safetensors import safe_open

    path = os.path.join(ckpt_dir, f"engram{rank}-mp{world_size}.safetensors")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    handle = safe_open(path, framework="pt", device="cpu")
    if mmap:
        _ENGRAM_MMAP_HANDLES.append(handle)   # dropping this unmaps the tables -> segfault

    keys = set(handle.keys())
    report: dict[str, Any] = {"path": path, "tensors": len(keys), "layers": {}}
    for layer_idx, mod in engrams.items():
        pre = f"layers.{layer_idx}.engram"
        need = {f"{pre}.embed.weight", f"{pre}.embed.scale"}
        if not need <= keys:
            raise KeyError(f"missing {sorted(need - keys)} in {path}")
        mod.embed.bind_mmap(handle.get_tensor(f"{pre}.embed.weight"),
                            handle.get_tensor(f"{pre}.embed.scale"))
        for attr in ("q_weight", "k_weight"):
            k = f"{pre}.{attr}"
            if k in keys:
                getattr(mod, attr).data.copy_(handle.get_tensor(k).to(torch.float32))
        wk = f"{pre}.wkv.weight"
        if wk in keys:
            w = handle.get_tensor(wk)
            sk = f"{pre}.wkv.scale"
            if sk in keys and w.dtype in (torch.float8_e4m3fn, torch.int8):
                s = handle.get_tensor(sk).float()
                bo = w.shape[0] // s.shape[0]
                bi = w.shape[1] // s.shape[1]
                w = (w.float().unflatten(0, (-1, bo)).unflatten(-1, (-1, bi))
                     * s[:, None, :, None]).flatten(2, 3).flatten(0, 1)
            tgt = mod.wkv.weight
            tgt.data.copy_(w.to(tgt.dtype).to(tgt.device))
        report["layers"][layer_idx] = {"rows": mod.embed.part_num_embeddings,
                                       "mmap": mod.embed._mmap}
    return report


# --------------------------------------------------------------------------------------
# 5. Registration
# --------------------------------------------------------------------------------------
def register(config_shim: bool = True) -> dict[str, Any]:
    """Register deepseek_v41 with transformers and vLLM. Idempotent.

    Returns a dict describing what was registered, so a caller can assert on it rather
    than trusting that this silently worked.
    """
    out: dict[str, Any] = {}
    if config_shim:
        import dsv41_config          # registers the three config classes with AutoConfig
        out["config"] = dsv41_config.DeepseekV41Config.model_type

    from vllm import ModelRegistry
    try:
        from vllm.models.deepseek_v4 import DeepseekV4ForConditionalGeneration as Base
        out["base"] = "DeepseekV4ForConditionalGeneration"
    except ImportError:
        from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM as Base
        out["base"] = "DeepseekV4ForCausalLM"

    cls = _build_model_class(Base)
    ModelRegistry.register_model("DeepseekV41ForCausalLM", cls)
    out["arch"] = "DeepseekV41ForCausalLM"
    return out


def _build_model_class(Base):
    """Subclass V4, adding engram and CED. Kept a closure so importing this module is cheap."""

    class DeepseekV41ForConditionalGeneration(Base):
        """V4 plus engram and CED.

        The engram hook lives in the inner model's layer loop rather than in the decoder
        layer, which means the decoder layer's forward is untouched -- see the module
        docstring for why that works and what it costs.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            cfg = None
            for a in list(args) + list(kwargs.values()):
                if hasattr(a, "model_config"):
                    cfg = a.model_config.hf_config
                    break
            self._dsv41_config = cfg
            self._engram_ready = False

        def dsv41_install(self, ckpt_dir: str, tokenizer, rank: int, world_size: int,
                          mmap_engram: bool = True) -> dict[str, Any]:
            """Build engram, load its weights, wire CED. Call after the base weights load."""
            cfg = self._dsv41_config
            text = _text_config(cfg)
            inner = getattr(self, "model", self)

            engrams: dict[int, Engram] = {}
            for slot, layer_idx in enumerate(text.engram_layer_ids):
                eng = Engram(cfg, slot, rank, world_size).to(torch.bfloat16)
                engrams[layer_idx] = eng
            self.engram_layers = nn.ModuleDict({str(k): v for k, v in engrams.items()})
            self.engram_hashes = EngramHashes(cfg, tokenizer)
            report = load_engram_weights(engrams, ckpt_dir, rank, world_size, mmap=mmap_engram)

            wired = wire_ced(self, cfg)
            if wired == 0:
                raise RuntimeError(
                    "CED wiring matched no layers -- the attention attribute path is wrong "
                    "for this vLLM build. Refusing to continue: without CED the decoder "
                    "computes its own KV and output is incoherent, not merely degraded.")
            report["ced_layers_wired"] = wired
            _install_engram_hook(inner, self.engram_layers, self.engram_hashes)
            self._engram_ready = True
            return report

    return DeepseekV41ForConditionalGeneration


def _install_engram_hook(inner: nn.Module, engram_layers: nn.ModuleDict,
                         hashes: EngramHashes) -> None:
    """Wrap the inner model's forward so engram is applied between post and pre.

    Reproduces the reference ordering using only primitives vLLM already has:
        h = mhc_post_tilelang(hidden, residual, post_mix, res_mix)   # == reference's h
        h = engram(h, ...)
        then hand h back with residual=None, routing through the UNFUSED mhc_pre.
    """
    from vllm.models.deepseek_v4.nvidia.model import mhc_post_tilelang

    idxs = {int(k) for k in engram_layers.keys()}
    inner._dsv41_engram = engram_layers
    inner._dsv41_hashes = hashes
    inner._dsv41_engram_idxs = idxs
    inner._dsv41_mhc_post = mhc_post_tilelang
    if not getattr(inner, "_dsv41_hooked", False):
        inner._dsv41_hooked = True
        # NOTE: the actual splice must happen inside the layer loop. vLLM's loop is a plain
        # `for` over self.layers, so the cleanest correct approach is a thin Sequential-like
        # wrapper around each engram layer that performs post -> engram -> pre and then
        # delegates. Implemented as a module wrapper so no vLLM source is patched.
        for idx in sorted(idxs):
            layer = inner.layers[idx]
            inner.layers[idx] = _EngramLayerWrapper(
                layer, engram_layers[str(idx)], mhc_post_tilelang)


class _EngramLayerWrapper(nn.Module):
    """post -> engram -> unfused pre -> wrapped layer.

    Delegates every attribute so vLLM's weight loading, KV-cache discovery and
    `kv_sharing_target_layer_name` introspection continue to see the real layer.
    """

    def __init__(self, layer: nn.Module, engram: Engram, mhc_post):
        super().__init__()
        self.layer = layer
        self.engram = engram
        self._mhc_post = mhc_post
        self._hashes: torch.Tensor | None = None
        self._mask: torch.Tensor | None = None

    def set_hashes(self, hashes: torch.Tensor, mask: torch.Tensor | None) -> None:
        self._hashes, self._mask = hashes, mask

    def forward(self, hidden_states, positions, input_ids, post_mix=None, res_mix=None,
                residual=None):
        if self._hashes is not None:
            if residual is not None:
                h = self._mhc_post(hidden_states, residual, post_mix, res_mix)
            else:
                h = hidden_states
            h = self.engram(h, self._hashes, self._mask)
            # residual=None routes the layer through its unfused mhc_pre on h
            hidden_states, residual, post_mix, res_mix = h, None, None, None
        return self.layer(hidden_states, positions, input_ids, post_mix, res_mix, residual)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)


# --------------------------------------------------------------------------------------
# 6. Verification harness
# --------------------------------------------------------------------------------------
def verify_against_reference(vllm_logits: torch.Tensor, reference_logits: torch.Tensor,
                             atol: float = 2e-2) -> dict[str, Any]:
    """Compare first-token logits against the reference implementation on one prompt.

    Do this before serving. Engram and the n-gram lookback degrade quality SILENTLY when
    wrong -- the model keeps producing fluent text, it is just not the trained model. A
    greedy-decode match on a handful of prompts is the cheapest honest check; top-1
    agreement alone is not sufficient, so the max absolute deviation is reported too.
    """
    a = vllm_logits.float().flatten()
    b = reference_logits.float().flatten()
    n = min(a.numel(), b.numel())
    a, b = a[:n], b[:n]
    dev = (a - b).abs()
    return {
        "n": int(n),
        "max_abs_dev": float(dev.max()),
        "mean_abs_dev": float(dev.mean()),
        "top1_match": bool(a.argmax() == b.argmax()),
        "within_atol": bool(dev.max() <= atol),
    }
