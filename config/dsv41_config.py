"""Register `deepseek_v41` so transformers and vLLM can both load the checkpoint.

    import dsv41_config            # registration happens on import

WHY THIS IS NEEDED
    No released transformers knows this architecture -- verified against 5.16.1 and 5.17.0,
    both of which ship deepseek_v2/v3/v32/v4 and no v41. Without a registered config,
    AutoConfig fails before vLLM reaches its own model registry.

WHY IT SUBCLASSES vLLM's CONFIG AND FLATTENS
    vLLM does NOT use the transformers config class for this family. It has its own
    `vllm.transformers_utils.configs.deepseek_v4.DeepseekV4Config`, and its multimodal
    preprocessor enforces it by identity:

        return self.ctx.get_hf_config(DeepseekV4Config)
        -> TypeError: Expected type ...DeepseekV4Config, but found ...DeepseekV41Config

    That class is deliberately thin: it declares the nine FLAT `vision_*` fields plus the
    rope parameters, and lets every text field arrive as a kwarg that PreTrainedConfig stores
    as an attribute. It also accepts `rope_scaling` AND `rope_parameters`, which is how V4
    ends up with the single flat yarn dict vLLM's max-len logic requires -- vLLM's own config
    performs that normalisation.

    So the correct shim is small: subclass vLLM's class, and FLATTEN V4.1's nested
    `text_config` / `vision_config` into the shape V4 already uses (text fields at top level,
    vision fields `vision_`-prefixed). That satisfies the isinstance check and every other V4
    code path at once, instead of one at a time.

    V4.1's genuinely new fields -- engram_*, kv_source_layer_ids, index_source_layer_ids,
    candidate_*, dspark_* -- need no declaration: PreTrainedConfig keeps unknown kwargs as
    attributes. They simply need a class to land on.

NOT A RENAME: `compress_ratios` vs `compress_rates`
    transformers types V4's `compress_rates` as `dict | None`; V4.1's `compress_ratios` is a
    per-LAYER list of length n_layers + 3 (43 for 40 layers, covering the MTP layers).
    Passing one as the other trips strict dataclass validation. They stay separate.
"""
from __future__ import annotations

from transformers import AutoConfig

__all__ = ["DeepseekV41Config", "register", "BASE_IS_VLLM"]

# Prefer vLLM's config class so the isinstance checks inside its V4 implementation pass.
# Fall back to transformers' when vLLM is not importable, so the config can be inspected
# and unit-tested without a vLLM install.
try:
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config as _Base
    BASE_IS_VLLM = True
except Exception:  # pragma: no cover - exercised only without vLLM
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import (
        DeepseekV4Config as _Base,
    )
    BASE_IS_VLLM = False


# V4.1 nests its vision encoder; V4 spells the same fields flat with a `vision_` prefix.
_VISION_MAP = {
    "num_hidden_layers": "vision_n_layers",
    "hidden_size": "vision_dim",
    "num_attention_heads": "vision_n_heads",
    "intermediate_size": "vision_inter_dim",
    "patch_size": "vision_patch_size",
    "rope_theta": "vision_rope_theta",
    "downsample_ratio": "vision_downsample_ratio",
    "max_image_tokens": "vision_max_n_token",
    "min_pixels": "vision_min_pixels",
    "max_wh_ratio": "vision_max_wh_ratio",
}


def _flatten(text_config, vision_config, kwargs: dict) -> dict:
    """Hoist nested sub-configs into the flat shape V4 uses."""
    flat: dict = {}
    if text_config:
        src = text_config if isinstance(text_config, dict) else text_config.to_dict()
        flat.update({k: v for k, v in src.items() if k != "model_type"})
    if vision_config:
        src = vision_config if isinstance(vision_config, dict) else vision_config.to_dict()
        for k, v in src.items():
            if k == "model_type":
                continue
            flat[_VISION_MAP.get(k, f"vision_{k}" if not k.startswith("vision_") else k)] = v
    # explicit top-level keys win over anything hoisted
    flat.update(kwargs)
    return flat


class DeepseekV41Config(_Base):
    """DeepSeek-V4.1-Flash, presented to vLLM in V4's flat layout."""

    model_type = "deepseek_v41"

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        flat = _flatten(text_config, vision_config, kwargs)

        # vLLM's V4 decoder gates hashed expert routing on a PREFIX rule:
        #     is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers
        # V4-Flash declares num_hash_layers: 3. V4.1 declares nothing, and the checkpoint
        # settles why -- it contains ZERO tid2eid/tie2eid tensors, while all 40 layers carry a
        # normal ffn.gate.*. So V4.1 does not use hashed routing at all: that mechanism was
        # REPLACED by engram (at layers [1, 14]), not extended by it. num_hash_layers must
        # therefore be 0, which disables the hash_moe path entirely.
        #
        # Do not be tempted to set this to len(engram_layer_ids): engram is not a prefix rule
        # and is not hashed routing, and conflating them would route two layers through the
        # wrong MoE implementation.
        flat.setdefault("num_hash_layers", 0)
        # keep the nested originals addressable for anything that prefers them, without
        # letting them reach the strict base __init__
        nested_text = flat.pop("_nested_text", None)
        super().__init__(**flat)
        self._nested_text = nested_text

    def get_text_config(self, decoder=False):
        """Flat layout: the text config IS this object, as it is for V4."""
        return self


def register() -> dict:
    """Idempotent. Returns what was registered so a caller can assert rather than hope."""
    try:
        AutoConfig.register(DeepseekV41Config.model_type, DeepseekV41Config)
    except ValueError:
        pass  # already registered
    return {
        "model_type": DeepseekV41Config.model_type,
        "base": _Base.__module__ + "." + _Base.__name__,
        "base_is_vllm": BASE_IS_VLLM,
    }


register()
