"""Stability test for the deepseek_v41 config registration.

Usage:  DSV41_CKPT=/path/to/DeepSeek-V4.1-Flash python tests/test_config_stability.py
Requires config/dsv41_config.py importable (e.g. PYTHONPATH=config).
"""
import json, sys, traceback

import os
CKPT = os.environ.get("DSV41_CKPT", "/path/to/DeepSeek-V4.1-Flash")
fails = []
def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (("  :: " + str(detail)) if detail else ""))
    if not cond:
        fails.append(name)

print("=== A. idempotent registration (import twice) ===")
import dsv41_config
import importlib; importlib.reload(dsv41_config)
import dsv41_config
check("double import/reload raises nothing", True)

from transformers import AutoConfig
from vllm import ModelRegistry
from vllm.models.deepseek_v4 import DeepseekV4ForConditionalGeneration as V4Cls
ModelRegistry.register_model("DeepseekV41ForCausalLM", V4Cls)
ModelRegistry.register_model("DeepseekV41ForCausalLM", V4Cls)   # twice
check("double model registration raises nothing", True)

print()
print("=== B. determinism: parse 5x, compare serialised configs ===")
from vllm.transformers_utils.config import get_config, is_rope_parameters_nested
sigs = []
for i in range(5):
    c = get_config(CKPT, trust_remote_code=False)
    t = c.get_text_config()
    sigs.append(json.dumps({
        "layers": t.num_hidden_layers, "hidden": t.hidden_size,
        "experts": t.n_routed_experts, "rope": t.rope_parameters,
        "engram": t.engram_num_embeddings, "kvsrc": t.kv_source_layer_ids,
    }, sort_keys=True))
check("5 parses identical", len(set(sigs)) == 1, f"{len(set(sigs))} distinct")

print()
print("=== C. rope shape matches what vLLM expects ===")
c = get_config(CKPT, trust_remote_code=False); t = c.get_text_config()
rp = t.rope_parameters
check("rope_parameters is flat with rope_type", isinstance(rp, dict) and "rope_type" in rp, rp)
check("rope_type == yarn", rp.get("rope_type") == "yarn")
check("factor == 16", rp.get("factor") == 16)
check("original_max_position_embeddings == 65536", rp.get("original_max_position_embeddings") == 65536)
check("rope_theta == 10000 (main, not compress 160000)", rp.get("rope_theta") == 10000, rp.get("rope_theta"))
check("compress theta still reachable", getattr(t, "compress_rope_theta", None) == 160000)

print()
print("=== D. ModelConfig resolves at several context lengths ===")
from vllm.config import ModelConfig
for mlen in (4096, 65536, 500000, None):
    try:
        mc = ModelConfig(model=CKPT, tokenizer=CKPT, dtype="bfloat16", seed=0,
                         max_model_len=mlen, trust_remote_code=False)
        check(f"max_model_len={mlen} -> resolved", True, f"arch={mc.architectures} got={mc.max_model_len}")
    except Exception as e:
        check(f"max_model_len={mlen} -> resolved", False, f"{type(e).__name__}: {str(e)[:160]}")

print()
print("=== E. derived context with no cap should be 1M (65536 x 16) ===")
try:
    mc = ModelConfig(model=CKPT, tokenizer=CKPT, dtype="bfloat16", seed=0,
                     max_model_len=None, trust_remote_code=False)
    check("uncapped max_model_len == 1048576", mc.max_model_len == 1048576, mc.max_model_len)
except Exception as e:
    check("uncapped resolves", False, str(e)[:160])

print()
print("=== F. structural fields intact ===")
exp = {"num_hidden_layers": 40, "hidden_size": 5120, "num_attention_heads": 64,
       "n_routed_experts": 384, "num_experts_per_tok": 6, "n_shared_experts": 1,
       "head_dim": 512, "index_n_heads": 32, "hc_mult": 4, "o_groups": 8,
       "engram_n_heads": 8, "engram_head_dim": 256, "dspark_n_routed_experts": 128}
for k, v in exp.items():
    check(f"{k} == {v}", getattr(t, k, None) == v, getattr(t, k, "MISSING"))
check("engram_layer_ids == [1,14]", list(t.engram_layer_ids) == [1, 14])
check("kv_source_layer_ids == [2,8,14,20]", list(t.kv_source_layer_ids) == [2, 8, 14, 20])
check("compress_ratios length == 43", len(t.compress_ratios) == 43, len(t.compress_ratios))
check("quantization expert_dtype == fp4",
      (getattr(c, "quantization_config", {}) or {}).get("expert_dtype") == "fp4")
check("TP=4 divides heads/experts", 64 % 4 == 0 and 384 % 4 == 0 and 128 % 4 == 0)

print()
print("=" * 52)
print(("ALL PASS" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
