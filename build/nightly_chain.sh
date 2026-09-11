#!/bin/bash
# nightly_chain.sh (build host): the reference's image lineage end to end -> vllm-dsv41:nightly
set -u; cd ${DSV41_HOME:-$HOME/dsv41}
NB=vllm/vllm-openai:nightly-8a728663c1c3eeace834a95f5654fa653cc1998c
PKG=/usr/local/lib/python3.12/dist-packages/vllm
stage(){ echo "=== $(date -u +%FT%TZ) $* ==="; }
die(){ echo "CHAIN_FAILED at $*"; echo "CHAIN_EXIT=1"; exit 1; }
stage PULL; docker pull -q $NB >/dev/null 2>&1 || die pull; docker images $NB --format "  {{.Size}}"
docker run --rm --entrypoint python3 $NB -c "import vllm,torch,flashinfer;print('  nightly: vllm',vllm.__version__,'torch',torch.__version__,'flashinfer',flashinfer.__version__)" 2>&1 | grep nightly
stage OVL1; B=build-ovl1; rm -rf $B; mkdir -p $B; cp -r dsv41-feat/vllm $B/vllm
printf 'FROM %s\nCOPY vllm/ %s/\nRUN find %s -name "__pycache__" -type d -prune -exec rm -rf {} + && python3 -c "import vllm; print(vllm.__version__)"\n' "$NB" "$PKG" "$PKG" > $B/Dockerfile
DOCKER_BUILDKIT=1 docker build -q -t vllm-dsv41:ovl1 $B >/dev/null 2>&1 || die ovl1; echo "  ovl1 ok"
stage EXTBUILD "(against the nightly's torch)"
find dsv41-feat/build -mindepth 1 -maxdepth 1 ! -name _deps -exec rm -rf {} + ; rm -rf dsv41-feat/build/_deps/cutlass-build dsv41-feat/build/_deps/cutlass-subbuild
docker rm -f v41build2 >/dev/null 2>&1; docker run -d --name v41build2 --memory 100g --memory-swap 100g --cpus 20 -v ${DSV41_HOME:-$HOME/dsv41}/dsv41-feat:/src --entrypoint sleep vllm-dsv41:ovl1 infinity >/dev/null
docker exec v41build2 bash -c 'export DEBIAN_FRONTEND=noninteractive; (apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1); pip install -q cmake ninja 2>/dev/null; which git cmake ninja | tr "\n" " "; echo'
docker cp ext_build_inner.sh v41build2:/tmp/inner.sh; docker exec v41build2 bash /tmp/inner.sh 2>&1 | grep -E "configure rc|build rc|BUILD_EXIT|FAILED|error:" | tail -4
SO=$(find dsv41-feat/build -maxdepth 1 -name "_C_stable_libtorch*.so" | head -1); [ -n "$SO" ] || die extbuild; docker rm -f v41build2 >/dev/null 2>&1
stage OVL1B; B=build-ovl1b; rm -rf $B; mkdir -p $B; cp "$SO" $B/_C_stable_libtorch.abi3.so
printf 'FROM vllm-dsv41:ovl1\nCOPY _C_stable_libtorch.abi3.so %s/_C_stable_libtorch.abi3.so\nRUN find %s -name "__pycache__" -type d -prune -exec rm -rf {} + ; python3 -c "import vllm"\n' "$PKG" "$PKG" > $B/Dockerfile
DOCKER_BUILDKIT=1 docker build -q -t vllm-dsv41:ovl1b $B >/dev/null 2>&1 || die ovl1b; echo "  ovl1b ok"
stage FI07; B=build-fi07n; rm -rf $B; mkdir -p $B; sed "s#^FROM .*#FROM vllm-dsv41:ovl1b#" build-fi07/Dockerfile > $B/Dockerfile
DOCKER_BUILDKIT=1 docker build -q -t vllm-dsv41:ovl3 $B >/tmp/fi07n.out 2>&1 || { tail -5 /tmp/fi07n.out; die fi07; }; echo "  ovl3 (FlashInfer 0.7.0rc1) ok"
stage PREWARM; B=build-v41n; rm -rf $B; mkdir -p $B; cp build-v41/prewarm5.py build-v41/verify5.py $B/; sed "s#^FROM .*#FROM vllm-dsv41:ovl3#" build-v41/Dockerfile > $B/Dockerfile
DOCKER_BUILDKIT=1 docker build -q -t vllm-dsv41:nightly $B >/tmp/v41n.out 2>&1 || { tail -5 /tmp/v41n.out; die prewarm; }; echo "  vllm-dsv41:nightly ok $(docker images -q vllm-dsv41:nightly | head -1 | cut -c1-12)"
stage SMOKE "(GPU: extension + rms_norm + model package + patch markers)"
PD=$HOME/patches/dsv41-boot10; M=""; while read -r f rel; do [ -z "$f" ] && continue; M="$M -v $PD/$f:$PKG/$rel:ro"; done < "$PD/mounts.txt"
docker run --rm --gpus all --entrypoint python3 -e DSV41_ENGRAM_DISK=1 $M vllm-dsv41:nightly -c "
import torch, vllm._C_stable_libtorch, inspect
from vllm import _custom_ops as ops
x=torch.randn(4,5120,device='cuda',dtype=torch.bfloat16); w=torch.ones(5120,device='cuda',dtype=torch.bfloat16); o=torch.empty_like(x); ops.rms_norm(o,x,w,1e-6); torch.cuda.synchronize(); print('  rms_norm OK finite=', bool(torch.isfinite(o).all()))
import vllm.models.deepseek_v4_1, vllm.models.deepseek_v4_1.common.engram as e, vllm.model_executor.model_loader.weight_utils as w, vllm.models.deepseek_v4_1.attention as a
print('  markers: DiskEngramTable', hasattr(e,'DiskEngramTable'), '| weight_utils', hasattr(w,'_DSV41_ENGRAM_DISK'), '| sm12x', 'sm12x' in inspect.getsource(a).lower())
import flashinfer; from flashinfer.mla import supported_sparse_mla_sm120_configs as f; print('  flashinfer', flashinfer.__version__, 'topk1152 decode', f()['dsv4'].supports_decode(num_heads=16, topk=1152))
" 2>&1 | grep -E "^  " 
echo "CHAIN_EXIT=0"
