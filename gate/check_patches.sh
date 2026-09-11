#!/bin/bash
# Do the bind-mounted patches actually reach what Python imports? No GPU, no model load.
SITE=/usr/local/lib/python3.12/dist-packages/vllm
PD=$HOME/patches/dsv41-boot10
M=""
while read -r f rel; do [ -z "$f" ] && continue; M="$M -v $PD/$f:$SITE/$rel:ro"; done < "$PD/mounts.txt"
cat > /tmp/chk.py <<'PY'
import os, inspect, vllm
print("vllm.__file__      =", vllm.__file__)
print("vllm.__version__   =", vllm.__version__)
import vllm.models.deepseek_v4_1.common.engram as eng
print("engram file        =", eng.__file__)
print("engram DiskEngramTable (patch marker):", hasattr(eng, "DiskEngramTable"))
print("engram _DSV41_ENGRAM_DISK            :", getattr(eng, "_DSV41_ENGRAM_DISK", "ATTR MISSING"))
import vllm.model_executor.model_loader.weight_utils as wu
print("weight_utils _DSV41_ENGRAM_DISK      :", getattr(wu, "_DSV41_ENGRAM_DISK", "ATTR MISSING => patch NOT applied"))
import vllm.models.deepseek_v4_1.attention as att
print("attention.py SM12x marker            :", "sm12x" in inspect.getsource(att).lower())
import vllm.model_executor.layers.sparse_attn_indexer as sai
print("indexer top_k_per_row_decode         :", "top_k_per_row_decode" in inspect.getsource(sai))
import vllm.models.deepseek_v4_1.nvidia.model_state as ms
print("model_state staged_rows (prestage)   :", "staged_rows" in inspect.getsource(ms))
PY
echo "=== WITH mounts + DSV41_ENGRAM_DISK=1 (what the launcher does) ==="
docker run --rm --entrypoint python3 -e DSV41_ENGRAM_DISK=1 $M -v /tmp/chk.py:/chk.py:ro ${IMAGE:-vllm-dsv41:branch} /chk.py 2>&1 | grep -vE "^(INFO|WARNING|W0|\[)" 
echo
echo "=== how far is the 0909 tree from the dsv41-feat patch files? (diff line counts, image-original vs patch) ==="
while read -r f rel; do [ -z "$f" ] && continue
  n=$(diff <(docker run --rm --entrypoint cat ${IMAGE:-vllm-dsv41:branch} "$SITE/$rel" 2>/dev/null) "$PD/$f" | grep -cE "^[<>]")
  printf "  %-24s diff-lines=%s\n" "$f" "$n"
done < "$PD/mounts.txt"
echo
echo "=== vllm serve flags (stderr merged this time) ==="
docker run --rm --entrypoint bash ${IMAGE:-vllm-dsv41:branch} -c 'vllm serve --help 2>&1 | grep -oE -- "--engram-config|--language-model-only|--speculative-config|--nnodes|--headless|--block-size|--tokenizer-mode" | sort -u | tr "\n" " "; echo; vllm serve --help 2>&1 | grep -A6 -- "--tokenizer-mode" | tr -s " " | grep -iE "deepseek|choices|\{" | head -3'
