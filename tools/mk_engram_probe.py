#!/usr/bin/env python3
"""Step-level NaN / engram probe generator for DeepSeek-V4.1 under CUDA graphs.

Why this exists: on a 4x GB10 fleet, graph mode produced output that every gate called healthy while ~1.7% of
decode steps carried NaN. No output-level check saw it. This generator builds three drop-in replacements that
instrument the forward with pure device ops (legal inside graph capture) and report eagerly between steps:

  engram_dbg.py      Engram.forward records the staged rows, the fp8 `wkv` output and the engram output
                     (NaN/inf count + abs-max), plus hash-id range and token count; keeps a copy of the
                     (possibly captured) `wkv` result so the report can diff it against an eager recompute.
                     EngramDiskStager.stage() also honours a control file so row CONTENT can be switched at
                     runtime without a restart: skip | zero | const | full (default full).
  model_dbg.py       per-layer hidden-state stats (slot 79 = the input to layer 0) and four per-decoder-layer
                     probes: attention output, post-attention residual, FFN input, FFN output.
  model_state_dbg.py the eager report, logged from prepare_inputs before staging: first bad layer, the sub-layer
                     stats around it, the engram lines, and KVCMP (captured vs eager `wkv`, maxdiff 0 = exact).

Requires PATCH_DIR to hold the reference patch set (engram.py, model_state.py) and BRANCH_MODEL_PY to point at
the model tree's nvidia/model.py. Writes mounts.extra.txt listing the three mounts; bind-mount them over the
originals, exclude the files they replace, and set DSV41_DBG=1.

Read the generated files before trusting them: the anchors are asserted unique, so a tree change fails loudly
rather than silently producing an inert probe (which is the failure mode that wastes a boot).
"""
import hashlib
import os
import pathlib

PAT = pathlib.Path(os.environ.get("PATCH_DIR", os.path.expanduser("~/patches/dsv41-boot10")))
BRANCH = pathlib.Path(os.environ.get("BRANCH_MODEL_PY", os.path.expanduser(
    "~/dsv41/dsv41-feat/vllm/models/deepseek_v4_1/nvidia/model.py")))
anchor = "import torch\n"


def rep1(s, old, new, what):
    assert s.count(old) == 1, f"anchor not unique/missing: {what} ({s.count(old)})"
    return s.replace(old, new, 1)


# ---------------- engram.py ----------------
src = PAT / "engram.py"
assert hashlib.md5(src.read_bytes()).hexdigest().startswith("c0329107"), "base is not the reference engram.py"
e = src.read_text(encoding="utf-8")
e = rep1(e, "        self.layer_hash_index = layer_hash_index\n",
         "        self.layer_hash_index = layer_hash_index\n"
         "        self._dsv41_layer_id = int(layout.layer_ids[layer_hash_index])\n", "engram init")
e = rep1(e, "        kv = self.wkv(self.embed(hash_ids).flatten(-2))\n        num_kv_tokens = hash_ids.shape[0]\n",
         "        _dsv41_rows = self.embed(hash_ids)\n"
         "        kv = self.wkv(_dsv41_rows.flatten(-2))\n"
         "        num_kv_tokens = hash_ids.shape[0]\n"
         "        if _DSV41_DBG_ON:\n"
         "            _dsv41_eng_record(self, 4, _dsv41_rows)\n"
         "            _dsv41_eng_record(self, 5, kv)\n"
         "            _dsv41_eng_hash(self, hash_ids)\n"
         "            _dsv41_eng_save_kv(self, kv)\n", "engram forward")
e = rep1(e, "            num_warps=num_warps,\n        )\n        return output\n",
         "            num_warps=num_warps,\n        )\n        if _DSV41_DBG_ON:\n            _dsv41_eng_record(self, 6, output)\n        return output\n",
         "engram output")
e = rep1(e, "        n = min(int(num_tokens), self.max_tokens)\n",
         "        if _dsv41_stage_mode() == \"skip\":\n            self.num_staged = 0\n            return 0\n"
         "        n = min(int(num_tokens), self.max_tokens)\n", "stage head")
e = rep1(e, "            engram.staged_rows[:n].copy_(staged, non_blocking=True)\n        self.num_staged = n\n",
         "            engram.staged_rows[:n].copy_(staged, non_blocking=True)\n"
         "        _dsv41_mode = _dsv41_stage_mode()\n"
         "        if _dsv41_mode == \"zero\":\n"
         "            for engram in self.engrams:\n                engram.staged_rows[:n].zero_()\n"
         "        elif _dsv41_mode == \"const\":\n"
         "            for engram in self.engrams:\n                engram.staged_rows[:n].fill_(0.01)\n"
         "        self.num_staged = n\n", "stage tail")
helpers_e = '''
import os as _dsv41_os
_DSV41_DBG_ON = _dsv41_os.environ.get("DSV41_DBG") == "1"
_DSV41_KV: dict = {}          # layer id -> persistent [64, N] copy of the (possibly captured) wkv output
_DSV41_MODE_FILE = _dsv41_os.environ.get("DSV41_STAGE_MODE_FILE", "/cache/dsv41_stage_mode")
_DSV41_MODE_LOGGED: set = set()


def _dsv41_stage_mode() -> str:
    try:
        with open(_DSV41_MODE_FILE) as f:
            mode = f.read().strip() or "full"
    except OSError:
        mode = "full"
    if mode not in _DSV41_MODE_LOGGED:
        _DSV41_MODE_LOGGED.add(mode)
        logger.warning("DSV41 stage mode -> %s", mode)
    return mode


def _dsv41_eng_buf(device):
    from vllm.models.deepseek_v4_1.nvidia import model as _m
    if not _m.DSV41_SUB:
        if torch.cuda.is_current_stream_capturing():
            return None
        _m.DSV41_SUB.append(torch.zeros(2, 10, 80, device=device, dtype=torch.float32))
    return _m.DSV41_SUB[0]


def _dsv41_eng_record(mod, stage, x):
    idx = getattr(mod, "_dsv41_layer_id", None)
    if idx is None or idx >= 80:
        return
    t = _dsv41_eng_buf(x.device)
    if t is None:
        return
    h = x.detach()
    h = h.float() if h.dtype != torch.float32 else h
    t[0, stage, idx].copy_((torch.isnan(h) | torch.isinf(h)).sum().to(torch.float32))
    t[1, stage, idx].copy_(h.abs().amax())


def _dsv41_eng_hash(mod, hash_ids):
    idx = getattr(mod, "_dsv41_layer_id", None)
    if idx is None or idx >= 80:
        return
    t = _dsv41_eng_buf(hash_ids.device)
    if t is None:
        return
    t[1, 7, idx].copy_(hash_ids.detach().amax().to(torch.float32))
    t[0, 7, idx].copy_((hash_ids.detach() < 0).sum().to(torch.float32))
    t[1, 8, idx].fill_(float(hash_ids.shape[0]))


def _dsv41_eng_save_kv(mod, kv):
    idx = getattr(mod, "_dsv41_layer_id", None)
    if idx is None:
        return
    buf = _DSV41_KV.get(idx)
    if buf is None:
        if torch.cuda.is_current_stream_capturing():
            return
        buf = torch.zeros(64, kv.shape[1], dtype=kv.dtype, device=kv.device)
        _DSV41_KV[idx] = buf
    m = min(kv.shape[0], 64)
    buf[:m].copy_(kv[:m])
'''
e = rep1(e, anchor, anchor + helpers_e, "engram torch import")
out_e = PAT / "engram_dbg.py"
out_e.write_text(e, encoding="utf-8")
compile(e, str(out_e), "exec")

# ---------------- model.py ----------------
m = BRANCH.read_text(encoding="utf-8")
call = (
    "            hidden_states, residual, post_mix, res_mix, pre_mix = layer(\n"
    "                hidden_states,\n                positions,\n                input_ids,\n                pre_mix,\n"
    "                post_mix,\n                res_mix,\n                residual,\n                engram_hashes,\n"
    "                engram_mask,\n            )\n"
)
m = rep1(m, call, call + "            if _DSV41_DBG_ON:\n                _dsv41_dbg_record(self, idx, hidden_states)\n", "layer call")
loop_head = "        for idx, layer in enumerate(\n            islice(self.layers, self.start_layer, self.end_layer),\n            start=self.start_layer,\n        ):\n"
m = rep1(m, loop_head, "        if _DSV41_DBG_ON:\n            _dsv41_dbg_record(self, 79, hidden_states)\n" + loop_head, "loop head")
m = rep1(m, "            layer_id = extract_layer_index(prefix)\n",
         "            layer_id = extract_layer_index(prefix)\n            self._dsv41_layer_idx = layer_id\n", "layer_id")
attn_line = "        x = self.attn(positions, x, None)\n"
m = rep1(m, attn_line, attn_line + "        if _DSV41_DBG_ON:\n            _dsv41_sub_record(self, 0, x)\n", "attn line")
post_pair = "        residual = mhc_post_tilelang(x, residual, post_mix, res_mix)\n        post_mix, res_mix, x, ffn_pre = mhc_pre_delayed_tilelang(\n"
m = rep1(m, post_pair,
         "        residual = mhc_post_tilelang(x, residual, post_mix, res_mix)\n"
         "        if _DSV41_DBG_ON:\n            _dsv41_sub_record(self, 1, residual)\n"
         "        post_mix, res_mix, x, ffn_pre = mhc_pre_delayed_tilelang(\n", "post pair")
ffn_pair = "        x = self.ffn(x, input_ids)\n        return x, residual, post_mix, res_mix, ffn_pre\n"
m = rep1(m, ffn_pair,
         "        if _DSV41_DBG_ON:\n            _dsv41_sub_record(self, 2, x)\n"
         "        x = self.ffn(x, input_ids)\n"
         "        if _DSV41_DBG_ON:\n            _dsv41_sub_record(self, 3, x)\n"
         "        return x, residual, post_mix, res_mix, ffn_pre\n", "ffn pair")
helpers_m = '''
import os as _dsv41_os
_DSV41_DBG_ON = _dsv41_os.environ.get("DSV41_DBG") == "1"
DSV41_DBG: dict = {}
DSV41_SUB: list = []   # [tensor[2, 10, 80]]: stages 0 attn_out 1 post_res 2 ffn_in 3 ffn_out | 4 rows 5 kv 6 eng_out 7 hash 8 T


def _dsv41_stats(h):
    h = h.detach()
    h = h.float() if h.dtype != torch.float32 else h
    return (torch.isnan(h) | torch.isinf(h)).sum().to(torch.float32), h.abs().amax()


def _dsv41_dbg_record(model, idx, hs):
    ent = DSV41_DBG.get(id(model))
    if ent is None:
        if torch.cuda.is_current_stream_capturing():
            return
        ent = (f"{model.__class__.__name__}[{len(model.layers)}L]", torch.zeros(2, 80, device=hs.device, dtype=torch.float32))
        DSV41_DBG[id(model)] = ent
    _, t = ent
    n, a = _dsv41_stats(hs)
    t[0, idx].copy_(n)
    t[1, idx].copy_(a)


def _dsv41_sub_record(layer, stage, x):
    idx = getattr(layer, "_dsv41_layer_idx", None)
    if idx is None or idx >= 80:
        return
    if not DSV41_SUB:
        if torch.cuda.is_current_stream_capturing():
            return
        DSV41_SUB.append(torch.zeros(2, 10, 80, device=x.device, dtype=torch.float32))
    t = DSV41_SUB[0]
    n, a = _dsv41_stats(x)
    t[0, stage, idx].copy_(n)
    t[1, stage, idx].copy_(a)
'''
m = rep1(m, anchor, anchor + helpers_m, "model torch import")
out_m = PAT / "model_dbg.py"
out_m.write_text(m, encoding="utf-8")
compile(m, str(out_m), "exec")

# ---------------- model_state.py ----------------
ms_src = PAT / "model_state.py"
assert hashlib.md5(ms_src.read_bytes()).hexdigest().startswith("0a14bee6"), "base is not the reference model_state.py"
s = ms_src.read_text(encoding="utf-8")
stage_line = next(l for l in s.split("\n") if "self.engram_stager.stage(" in l)
indent = stage_line[: len(stage_line) - len(stage_line.lstrip())]
s = rep1(s, stage_line + "\n", indent + "if _DSV41_DBG_ON:\n" + indent + "    _dsv41_dbg_report(self)\n" + stage_line + "\n", "stage call")
helpers_s = '''
import os as _dsv41_os
from vllm.logger import init_logger as _dsv41_init_logger
_dsv41_logger = _dsv41_init_logger(__name__)
_DSV41_DBG_ON = _dsv41_os.environ.get("DSV41_DBG") == "1"
_DSV41_DBG_STEP = [0]
_DSV41_STAGES = ("attn_out", "post_res", "ffn_in", "ffn_out")


def _dsv41_kvcmp(state, sub):
    """Eager recompute of each engram layer's wkv on the SAME staged rows the last forward used; compare with the
    captured kv saved by engram_dbg. Runs on every rank unconditionally (embed() may all-gather)."""
    from vllm.models.deepseek_v4_1.common import engram as _e
    stager = state.engram_stager
    kvd = getattr(_e, "_DSV41_KV", None)
    if stager is None or not kvd or sub is None:
        return ""
    out = ""
    for engram in stager.engrams:
        lid = engram._dsv41_layer_id
        cap = kvd.get(lid)
        T = int(sub[1, 8, lid])
        if cap is None or T <= 0:
            continue
        with torch.inference_mode():
            rows = engram.embed(torch.empty(T, device=cap.device))
            ref = engram.wkv(rows.flatten(-2))
        mm = min(T, ref.shape[0], cap.shape[0])
        c = cap[:mm].float()
        r = ref[:mm].float()
        d = (c - r).abs()
        out += " || KVCMP L%d T=%d m=%d cap:nan=%d,max=%.3g ref:nan=%d,max=%.3g maxdiff=%.3g" % (
            lid, T, mm, int((torch.isnan(c) | torch.isinf(c)).sum()), float(c.abs().amax()),
            int((torch.isnan(r) | torch.isinf(r)).sum()), float(r.abs().amax()), float(d.nan_to_num(nan=1e30).amax()))
    return out


def _dsv41_dbg_report(state):
    from vllm.models.deepseek_v4_1.nvidia import model as _m
    _DSV41_DBG_STEP[0] += 1
    step = _DSV41_DBG_STEP[0]
    sub = _m.DSV41_SUB[0].cpu() if _m.DSV41_SUB else None
    kvcmp = _dsv41_kvcmp(state, sub)
    for name, t in _m.DSV41_DBG.values():
        c = t.cpu()
        bad = [b for b in (c[0] > 0).nonzero().flatten().tolist() if b != 79]
        if bad or step <= 4 or (sub is not None and float(sub[1, 8, 1]) > 0):
            msg = "DSV41_DBG step=%d %s INPUT nan=%d absmax=%.2f | first_bad_layer=%s bad=%s" % (
                step, name, int(c[0, 79]), float(c[1, 79]), bad[0] if bad else None, bad[:8])
            if sub is not None:
                if bad:
                    L = bad[0]
                    for lay in ([L - 1] if L > 0 else []) + [L]:
                        st = " ".join("%s:nan=%d,max=%.2f" % (_DSV41_STAGES[k], int(sub[0, k, lay]), float(sub[1, k, lay])) for k in range(4))
                        msg += " || layer%d: %s" % (lay, st)
                for lay in (1, 14):
                    msg += " || ENGRAM L%d T=%d rows:nan=%d,max=%.3g kv:nan=%d,max=%.3g out:nan=%d,max=%.3g hash:max=%d,neg=%d" % (
                        lay, int(sub[1, 8, lay]), int(sub[0, 4, lay]), float(sub[1, 4, lay]), int(sub[0, 5, lay]), float(sub[1, 5, lay]),
                        int(sub[0, 6, lay]), float(sub[1, 6, lay]), int(sub[1, 7, lay]), int(sub[0, 7, lay]))
            msg += kvcmp
            _dsv41_logger.warning(msg)
        t.zero_()
    if sub is not None:
        _m.DSV41_SUB[0].zero_()
'''
first_import = s.index("import ")
line_end = s.index("\n", first_import) + 1
s = s[:line_end] + helpers_s + s[line_end:]
out_s = PAT / "model_state_dbg.py"
out_s.write_text(s, encoding="utf-8")
compile(s, str(out_s), "exec")
(PAT / "mounts.extra.txt").write_text(
    "engram_dbg.py models/deepseek_v4_1/common/engram.py\n"
    "model_dbg.py models/deepseek_v4_1/nvidia/model.py\n"
    "model_state_dbg.py models/deepseek_v4_1/nvidia/model_state.py\n", encoding="utf-8")
print("engram probe v2:", hashlib.md5(out_e.read_bytes()).hexdigest()[:8], hashlib.md5(out_m.read_bytes()).hexdigest()[:8],
      hashlib.md5(out_s.read_bytes()).hexdigest()[:8], "| manifest written")
