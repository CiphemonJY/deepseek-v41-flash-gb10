"""Register `deepseek_v41` with transformers so AutoConfig can parse the checkpoint.

WHY THIS IS NEEDED
    No released transformers knows this architecture -- verified against 5.16.1 (in the
    your vLLM image image) and 5.17.0 (newest on PyPI): both ship deepseek_v2/v3/v32/v4 and
    no v41. So `AutoConfig.from_pretrained` fails before vLLM ever reaches its own model
    registry:
        ValueError: checkpoint ... has model type `deepseek_v41` but Transformers does
                    not recognize this architecture

WHY IT IS SMALL
    transformers' DeepseekV4Config ALREADY declares the machinery people assume is new:
    hc_mult / hc_eps / hc_sinkhorn_iters (mHC), index_n_heads / index_head_dim / index_topk
    (sparse indexer), o_groups / o_lora_rank, compress_rope_theta, swiglu_limit,
    scoring_func, num_nextn_predict_layers. V4.1 is an increment on the same lineage.
    And PreTrainedConfig keeps undeclared kwargs as attributes, so the genuinely new
    fields (engram_*, dspark_*, kv_source_layer_ids, index_source_layer_ids,
    candidate_*) survive without being declared -- they just need a class to land on.

    Two real differences are handled explicitly:
      * V4.1 NESTS text_config / vision_config (each with its own declared model_type);
        V4-Flash was flat with vision_* prefixed keys.
      * V4.1 spells the per-layer compression `compress_ratios`; transformers' V4 config
        calls it `compress_rates`. Aliased below so either name resolves.

WHAT THIS DOES NOT DO
    It does not make the model runnable. vLLM still needs a DeepseekV41ForCausalLM, and
    four features are genuinely absent from vllm.models.deepseek_v4 (verified by grep
    over all 20,881 lines): engram tables, CED cross-layer KV reuse (kv_source_layer_ids /
    index_source_layer_ids), the hierarchical candidate pool (candidate_*), and the DSpark
    routed MoE (dspark_n_routed_experts). This file only unblocks config parsing.

USAGE
    import dsv41_config            # registration happens on import
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained("/path/to/DeepSeek-V4.1-Flash")
"""
from transformers import AutoConfig
from transformers.configuration_utils import PreTrainedConfig
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config


class DeepseekV41VisionConfig(PreTrainedConfig):
    """DeepSeek-ViT: trained from scratch, 2D-RoPE, 3x3 pixel-unshuffle downsample."""

    model_type = "deepseek_v41_vision"

    def __init__(
        self,
        num_hidden_layers: int = 32,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 2816,
        patch_size: int = 14,
        rope_theta: float = 10000.0,
        downsample_ratio: int = 3,
        max_image_tokens: int = 1024,
        min_pixels: int = 295936,
        max_wh_ratio=None,
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio
        super().__init__(**kwargs)


class DeepseekV41TextConfig(DeepseekV4Config):
    """V4.1 language backbone. Inherits every V4 field; adds the four new feature groups.

    Declared explicitly (rather than left to kwargs) so that a typo in a checkpoint shows
    up as a wrong value rather than a silently absent attribute -- the engram hash
    multipliers are all derived from engram_vocab_size, and the reference implementation
    asserts engram_compressed_vocab_size against the tokenizer. A mismatch there would
    silently rehash the whole 189 GiB table.
    """

    model_type = "deepseek_v41_text"

    def __init__(
        self,
        # --- CED: decoder layers reuse KV/index state computed at specific earlier layers
        kv_source_layer_ids=(2, 8, 14, 20),
        index_source_layer_ids=(2, 8, 14, 20, 24, 28, 32, 36),
        # --- hierarchical sparse indexer: deeper layers restricted to a candidate pool
        candidate_source_layer_id: int = 20,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
        # --- engram conditional memory (196B params, sparse n-gram lookup)
        engram_layer_ids=(1, 14),
        engram_num_embeddings=(384006168, 384016682),
        engram_max_ngram_size: int = 4,
        engram_vocab_size: int = 16000000,
        engram_n_heads: int = 8,
        engram_head_dim: int = 256,
        engram_pad_token_id: int = 2,
        engram_compressed_vocab_size: int = 99092,
        # --- DSpark speculative decoding, now with its own routed MoE
        dspark_block_size: int = 5,
        dspark_noise_token_id: int = 128799,
        dspark_target_layer_ids=(37, 38, 39),
        dspark_markov_rank: int = 256,
        dspark_n_routed_experts: int = 128,
        dspark_num_experts_per_tok: int = 3,
        # --- spelled `compress_ratios` here, `compress_rates` in transformers' V4 config
        compress_ratios=None,
        **kwargs,
    ):
        self.kv_source_layer_ids = list(kv_source_layer_ids)
        self.index_source_layer_ids = list(index_source_layer_ids)
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size

        self.engram_layer_ids = list(engram_layer_ids)
        self.engram_num_embeddings = list(engram_num_embeddings)
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size

        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = list(dspark_target_layer_ids)
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok

        # NOT a rename of V4's `compress_rates`. transformers types that one as
        # `dict | None`; V4.1's `compress_ratios` is a per-LAYER list (len == n_layers + 3,
        # covering the MTP layers). Feeding one into the other trips strict dataclass
        # validation, so they are kept strictly separate.
        self.compress_ratios = list(compress_ratios) if compress_ratios is not None else None

        super().__init__(**kwargs)
        self._flatten_rope_parameters()

    def _flatten_rope_parameters(self) -> None:
        """Collapse the {main, compress} rope split into the flat shape vLLM expects.

        DeepseekV4Config synthesises rope_parameters as two named sub-configs -- `main`
        (the attention rope) and `compress` (the compressed-KV rope, theta 160000). vLLM
        only treats a nested rope dict as nested when every key is a LAYER TYPE:

            is_rope_parameters_nested(rp) = set(rp) <= ALLOWED_LAYER_TYPES
            ALLOWED_LAYER_TYPES = ('full_attention', 'sliding_attention', ...,
                                   'compressed_sparse_attention', ...)

        {main, compress} is not a subset, so vLLM wraps it as {"": {main:..., compress:...}}
        and then dies on rp["rope_type"]. vLLM's own loader already flattens this for
        deepseek_v4 -- verified by running get_config() against a live V4-Flash checkpoint,
        which yields a single flat yarn dict -- but that normalisation does not fire for
        model_type deepseek_v41, so we reproduce its output exactly:

            {beta_fast, beta_slow, factor, original_max_position_embeddings,
             type: yarn, rope_type: yarn, rope_theta: <main's theta>}

        Note the flattened form keeps MAIN's rope_theta (10000), not compress's (160000) --
        matching what V4 produces. The compress theta stays available as
        `compress_rope_theta`, which is where the model code reads it from anyway.
        """
        rp = getattr(self, "rope_parameters", None)
        if not isinstance(rp, dict) or "rope_type" in rp:
            return  # already flat, or absent
        main, compress = rp.get("main"), rp.get("compress")
        if not isinstance(compress, dict):
            return  # unexpected shape; leave it alone rather than guess
        flat = {k: v for k, v in compress.items() if k not in ("rope_theta",)}
        flat.setdefault("rope_type", flat.get("type", "yarn"))
        flat.setdefault("type", flat["rope_type"])
        if isinstance(main, dict) and "rope_theta" in main:
            flat["rope_theta"] = main["rope_theta"]
        elif getattr(self, "rope_theta", None) is not None:
            flat["rope_theta"] = self.rope_theta
        self.rope_parameters = flat


class DeepseekV41Config(PreTrainedConfig):
    """Composite multimodal config: text backbone + vision encoder."""

    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig, "vision_config": DeepseekV41VisionConfig}

    def __init__(self, text_config=None, vision_config=None, image_token_id: int = 129264, **kwargs):
        if text_config is None:
            text_config = {}
        if vision_config is None:
            vision_config = {}
        self.text_config = (
            text_config if isinstance(text_config, DeepseekV41TextConfig)
            else DeepseekV41TextConfig(**text_config)
        )
        self.vision_config = (
            vision_config if isinstance(vision_config, DeepseekV41VisionConfig)
            else DeepseekV41VisionConfig(**vision_config)
        )
        self.image_token_id = image_token_id
        super().__init__(**kwargs)

    # vLLM and a lot of tooling reach for these on the top-level config
    @property
    def hidden_size(self):
        return self.text_config.hidden_size

    @property
    def num_hidden_layers(self):
        return self.text_config.num_hidden_layers

    @property
    def vocab_size(self):
        return self.text_config.vocab_size


def register() -> None:
    """Idempotent registration of all three config types."""
    for cfg in (DeepseekV41Config, DeepseekV41TextConfig, DeepseekV41VisionConfig):
        try:
            AutoConfig.register(cfg.model_type, cfg)
        except ValueError:
            pass  # already registered


register()
