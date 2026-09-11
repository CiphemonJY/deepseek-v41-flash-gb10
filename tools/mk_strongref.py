#!/usr/bin/env python3
"""Strong-ref patch for breakable_cudagraph.py (council kernel seat, H1 mitigation ii).
Keep a strong reference to each captured entry's output so the shared graph pool cannot reclaim the slot
while an eager consumer (e.g. the MXFP8 SM120 GEMM in prepare_inputs) still reads it at replay."""
import os, pathlib, hashlib, sys
src = pathlib.Path(os.path.expanduser("~/dsv41/dsv41-feat/vllm/compilation/breakable_cudagraph.py"))
dst = pathlib.Path(os.path.expanduser("~/patches/dsv41-boot10/breakable_cudagraph.py"))
s = src.read_text(encoding="utf-8")
a = "    output: Any = None\n    input_addresses: list[int] | None = None\n"
b = "    output: Any = None\n    output_strong: Any = None  # DSV41 strong-ref patch: pins the pool slot(s) the output lives in\n    input_addresses: list[int] | None = None\n"
assert s.count(a) == 1, "dataclass anchor"
s = s.replace(a, b)
c = "            get_offloader().join_after_forward()\n"
d = c + "            # DSV41 strong-ref patch: hold the strong ref so the cudagraph pool cannot reuse this\n            # output's memory for the next descriptor; eager segments between graphs consume it at replay.\n            entry.output_strong = output\n"
assert s.count(c) == 1, "capture anchor"
s = s.replace(c, d)
dst.write_text(s, encoding="utf-8")
print("patched:", dst, "md5", hashlib.md5(dst.read_bytes()).hexdigest()[:8], "| orig md5", hashlib.md5(src.read_bytes()).hexdigest()[:8])
compile(s, str(dst), "exec"); print("syntax OK")
