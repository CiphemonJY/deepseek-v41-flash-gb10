#!/usr/bin/env bash
# launch41.sh <rank> -- DeepSeek-V4.1-Flash TP4 on THIS fleet via vLLM (dsv41-feat) + tonyd2wild's SM12x patches.
# Adapted from tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark launch/dsv41-tp4.sh (same structure and knobs),
# with this fleet's fabric, paths, port and model names. DROP-IN for the previous 4.0 serve:
#   port 8888, old served name first, so the :8000 proxy, both watchdogs and external callers need no change.
#
# Knobs (export before running, SAME on all four):
#   IMAGE       (default vllm-dsv41:fi07)      GMU (0.80)   MAXLEN (131072)   SEQS (8)   MAX_BATCHED (8192)
#   EAGER       1 (default) => --enforce-eager ; 0 => CUDA graphs (needs model_state.py prestage patch; sizes auto)
#   SPEC        none (default) | dspark (k=5, adaptive verification OFF: padded rows hang SM120 sparse MLA, FlashInfer #5015)
#   TEXT_ONLY   1 (default) => --language-model-only ; PARSERS 0|1 ; THINKING false
#   PATCH_NAME  dsv41-boot10 (default) -> ~/patches/$PATCH_NAME/{mounts.txt,...}
set -euo pipefail
NODE_RANK="${1:?usage: launch41.sh <0|1|2|3>}"
# gmu 0.72 (not 0.80): leaves ~34GiB physically free for the load-time page-cache transient.
# gmu 0.80 OOM-rebooted two nodes (40x NV_ERR_NO_MEMORY during weight load).
IMAGE="${IMAGE:-vllm-dsv41:branch}"; GMU="${GMU:-0.72}"; MAXLEN="${MAXLEN:-131072}"; SEQS="${SEQS:-8}"
MAX_BATCHED="${MAX_BATCHED:-8192}"; EAGER="${EAGER:-1}"; CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}"
CG_SIZES="${CG_SIZES:-}"; SPEC="${SPEC:-none}"; SPEC_K="${SPEC_K:-5}"; TEXT_ONLY="${TEXT_ONLY:-1}"
THINKING="${THINKING:-false}"; PARSERS="${PARSERS:-0}"; VLLM_EXTRA="${VLLM_EXTRA:-}"
PATCH_NAME="${PATCH_NAME:-dsv41-boot10}"

NAME="vllm_dsv41"
MODEL_HOST="${DSV41_HOME:-$HOME/dsv41}/hf-ckpt"          # full checkpoint is LOCAL on every rank (no NFS needed)
CACHE_HOST="${DSV41_HOME:-$HOME/dsv41}/vllm-cache"
SITE="/usr/local/lib/python3.12/dist-packages/vllm"
HEAD_IP="${HEAD_IP:?set HEAD_IP=<rank0 fabric ip>}"; MPORT="29541"; PORT="8888"
case "$NODE_RANK" in
  0) HOST_IP="${RANK0_IP:?set RANK0_IP}"; HEADLESS="" ;;
  1) HOST_IP="${RANK1_IP:?set RANK1_IP}"; HEADLESS="--headless" ;;
  2) HOST_IP="${RANK2_IP:?set RANK2_IP}"; HEADLESS="--headless" ;;
  3) HOST_IP="${RANK3_IP:?set RANK3_IP}"; HEADLESS="--headless" ;;
  *) echo "rank must be 0-3" >&2; exit 2 ;;
esac
# the 0909 tag needs the tokenizer mode spelled out; the branch image resolves it from model_type
# --tokenizer-mode deepseek_v41: NEEDED by the 0909-tree images (0909 tag, :fi07, :v41), REJECTED by the
# dsv41-feat tree (:branch, their overlay) whose CLI choices lack the literal and auto-resolve from model_type.
TOK_ARGS=""; case "$IMAGE" in *deepseekv41-flash-0909*|vllm-dsv41:fi07|vllm-dsv41:v41) TOK_ARGS="--tokenizer-mode deepseek_v41" ;; esac
# RoCE rails: set NCCL_IB_HCA / FABRIC_IFACES / FABRIC_IFACE0 / FABRIC_SUBNET for your nodes (see README)
HCA="${NCCL_IB_HCA:?set NCCL_IB_HCA to your RoCE HCA list, e.g. mlx5_1:1,mlx5_3:1}"

# ---- preflight ----
test -f "$MODEL_HOST/config.json" || { echo "MODEL MISSING at $MODEL_HOST" >&2; exit 3; }
test -f "$MODEL_HOST/model-00048-of-00048.safetensors" || { echo "MODEL INCOMPLETE (shard 48)" >&2; exit 3; }
PATCH_DIR="$HOME/patches/$PATCH_NAME"; test -f "$PATCH_DIR/mounts.txt" || { echo "no $PATCH_DIR/mounts.txt" >&2; exit 3; }
PATCH_MOUNTS=""
while read -r f rel; do
  [ -z "$f" ] && continue
  test -f "$PATCH_DIR/$f" || { echo "PATCH FILE MISSING: $PATCH_DIR/$f" >&2; exit 3; }
  PATCH_MOUNTS="$PATCH_MOUNTS -v $PATCH_DIR/$f:$SITE/$rel:ro"
done < "$PATCH_DIR/mounts.txt"
# extra (experimental) mounts: "<file> <site-relative path>" per line in mounts.extra.txt -- not md5-gated, logged
if [ -f "$PATCH_DIR/mounts.extra.txt" ]; then
  while read -r f rel; do [ -z "$f" ] && continue; test -f "$PATCH_DIR/$f" || { echo "EXTRA PATCH MISSING: $PATCH_DIR/$f" >&2; exit 3; }
    PATCH_MOUNTS="$PATCH_MOUNTS -v $PATCH_DIR/$f:$SITE/$rel:ro"; EXTRA_PATCHES="${EXTRA_PATCHES:-}$f($(md5sum "$PATCH_DIR/$f" | cut -c1-8)) "; done < "$PATCH_DIR/mounts.extra.txt"
fi
if [ "$EAGER" != "1" ] && ! grep -q '^model_state.py ' "$PATCH_DIR/mounts.txt"; then
  echo "EAGER=0 needs the Engram prestage patch (model_state.py)" >&2; exit 3; fi
mkdir -p "$CACHE_HOST"

# ---- graphs / speculation / mode args ----
GRAPH_ENV=""
if [ "$EAGER" = "1" ]; then GRAPH_ARGS=(--enforce-eager); SPEC_ADAPT=false
else
  if [ -z "$CG_SIZES" ]; then
    if [ "$SPEC" = "dspark" ]; then CG_SIZES=$( { seq "$SPEC_K" "$SPEC_K" $((SPEC_K * SEQS)); seq $((SPEC_K + 1)) $((SPEC_K + 1)) $(((SPEC_K + 1) * SEQS)); } | sort -n -u | paste -sd, - )
    else CG_SIZES=$(seq 1 "$SEQS" | paste -sd, -); fi
  fi
  GRAPH_ARGS=(--compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"cudagraph_capture_sizes\":[$CG_SIZES]}")
  GRAPH_ENV="-e VLLM_USE_BREAKABLE_CUDAGRAPH=1"; SPEC_ADAPT="${SPEC_ADAPT:-false}"
fi
if [ "$SPEC" = "dspark" ]; then
  SPEC_ARGS="--speculative-config {\"method\":\"dspark\",\"num_speculative_tokens\":$SPEC_K,\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":$SPEC_ADAPT}"
else SPEC_ARGS=""; fi
[ "$TEXT_ONLY" = "1" ] && TEXT_ARGS="--language-model-only" || TEXT_ARGS=""
[ "$PARSERS" = "1" ] && PARSER_ARGS="--tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41" || PARSER_ARGS=""

# ---- PRE-LAUNCH GATE (no GPU): refuse to launch the launch that OOM-rebooted the fleet 3x on 2026-09-11 ----
# a) every staged patch must match the reference's own md5 table (patch/README.md) and carry no CR bytes
declare -A REF_MD5=( [engram.py]=c0329107 [model_state.py]=0a14bee6 [weight_utils.py]=7e1027f1 [attention.py]=da9ef196
                     [flashinfer_sparse.py]=af0f8447 [sparse_swa.py]=cc419353 [sparse_attn_indexer.py]=a9b73756 [mounts.txt]=79a774bc )
for f in "${!REF_MD5[@]}"; do
  got=$(md5sum "$PATCH_DIR/$f" | cut -c1-8); [ "$got" = "${REF_MD5[$f]}" ] || { echo "GATE FAIL: $f md5 $got != reference ${REF_MD5[$f]} (CRLF? stale?)" >&2; exit 5; }
  [ "$(grep -c $'\r' "$PATCH_DIR/$f")" = 0 ] || { echo "GATE FAIL: $f has CR bytes" >&2; exit 5; }
done
# b) inside the exact image+mounts+env: the package imports and every patch's marker symbol is live
GATE_OUT=$(docker run --rm --entrypoint python3 -e DSV41_ENGRAM_DISK=1 $PATCH_MOUNTS "$IMAGE" -c '
import inspect, vllm.models.deepseek_v4_1
import vllm.models.deepseek_v4_1.common.engram as e, vllm.model_executor.model_loader.weight_utils as w
import vllm.models.deepseek_v4_1.attention as a, vllm.models.deepseek_v4_1.nvidia.model_state as ms
import vllm.model_executor.layers.sparse_attn_indexer as si
ok = hasattr(e,"DiskEngramTable") and getattr(e,"_DSV41_ENGRAM_DISK",False) and getattr(w,"_DSV41_ENGRAM_DISK",False) \
     and "sm12x" in inspect.getsource(a).lower() and "EngramDiskStager" in inspect.getsource(ms) and "top_k_per_row_decode" in inspect.getsource(si)
print("GATE_OK" if ok else "GATE_MARKERS_MISSING")' 2>&1 | grep -vE "^(INFO|WARNING|W0|\[)" | tail -1)
[ "$GATE_OUT" = "GATE_OK" ] || { echo "GATE FAIL: patch markers not live in $IMAGE -> $GATE_OUT" >&2; exit 5; }
echo "  pre-launch gate: md5 8/8 vs reference, no CR, all patch markers live in $IMAGE"
[ "${GATE_ONLY:-0}" = "1" ] && { echo "  GATE_ONLY=1: stopping before docker run (rank $NODE_RANK, image $IMAGE, avail=$(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 ))GiB)"; exit 0; }
# only now (gate passed, not a dry run) is it safe to remove a previous container on this rank
docker rm -f "$NAME" 2>/dev/null || true
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
AVAIL_GB=$(( $(grep MemAvailable /proc/meminfo | awk '{print $2}') / 1048576 ))
[ "$AVAIL_GB" -ge 100 ] || { echo "MemAvailable ${AVAIL_GB} GiB < 100 GiB, refusing to boot" >&2; exit 4; }

# ---- NCCL profile: legacy = the previous serve's dual-rail/LL128 env (proven in eager); ref = tonyd2wild plain env on our NIC names ----
HCA0="${HCA%%,*}"; IF0="${FABRIC_IFACE0:?}"
if [ "${NCCL_ENV_MODE:-legacy}" = "ref" ]; then
  NCCL_ARGS="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=$HCA0 -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=${FABRIC_SUBNET:?fabric CIDR, e.g. 10.10.0.0/24} -e NCCL_SOCKET_IFNAME=$IF0 -e GLOO_SOCKET_IFNAME=$IF0 -e TP_SOCKET_IFNAME=$IF0 -e MN_IF_NAME=$IF0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
  for kv in ${NCCL_SET:-}; do NCCL_ARGS="$NCCL_ARGS -e $kv"; done
else
  NCCL_ARGS="-e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=$HCA" -e NCCL_SOCKET_IFNAME=${FABRIC_IFACES:?set FABRIC_IFACES=<comma list>} -e GLOO_SOCKET_IFNAME=${FABRIC_IFACE0:?} -e NCCL_MAX_NCHANNELS=4 -e NCCL_MIN_NCHANNELS=4 -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e NCCL_IB_TC=106 -e NCCL_NET_PLUGIN=none -e NCCL_IB_MERGE_NICS=0 -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_PROTO=LL128 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
  # NCCL_DROP="PROTO CROSS_NIC ..." removes -e NCCL_<NAME>=... from the legacy set; NCCL_SET="K=V ..." appends overrides
  for n in ${NCCL_DROP:-}; do NCCL_ARGS=$(printf "%s" "$NCCL_ARGS" | sed -E "s/-e NCCL_${n}=[^ ]+ ?//"); done
  for kv in ${NCCL_SET:-}; do NCCL_ARGS="$NCCL_ARGS -e $kv"; done
fi

EXTRA_ENV_ARGS=""; for kv in ${EXTRA_ENV:-}; do EXTRA_ENV_ARGS="$EXTRA_ENV_ARGS -e $kv"; done
# --memory 112g caps the container BELOW the 121.7 GiB unified pool: an overrun is a contained
# container OOM, not a kernel OOM-kill of unrelated services / a hard reboot (2026-09-11).
# shellcheck disable=SC2086
docker run --gpus all -d --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g --memory 112g --memory-swap 112g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband \
  --oom-score-adj 500 \
  -v "$MODEL_HOST:/models/DeepSeek-V4.1-Flash:ro" \
  -v "$CACHE_HOST:/cache" \
  $PATCH_MOUNTS \
  -e VLLM_HOST_IP=$HOST_IP -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm${CACHE_TAG:+-$CACHE_TAG} -e TILELANG_CACHE_DIR=/cache/tilelang${CACHE_TAG:+-$CACHE_TAG} -e TRITON_CACHE_DIR=/cache/triton${CACHE_TAG:+-$CACHE_TAG} \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e "PYTORCH_CUDA_ALLOC_CONF=${ALLOC_CONF:-expandable_segments:True}" -e "VLLM_LOGGING_LEVEL=${LOG_LEVEL:-INFO}" $EXTRA_ENV_ARGS \
  -e VLLM_USE_RUST_FRONTEND=0 -e VLLM_HAS_FLASHINFER_CUBIN=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 \
  -e DSV41_ENGRAM_DISK=1 -e DSV41_ENGRAM_DISK_THREADS=32 -e DSV41_ENGRAM_DISK_CHUNK=16 \
  $GRAPH_ENV \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  $NCCL_ARGS \
  "$IMAGE" \
    /models/DeepSeek-V4.1-Flash \
    --served-model-name ${SERVED_NAMES:-deepseek-v41-flash} --host 0.0.0.0 --port "$PORT" \
    $TOK_ARGS \
    --tensor-parallel-size 4 --gpu-memory-utilization "$GMU" --max-model-len "$MAXLEN" \
    --max-num-seqs "$SEQS" --max-num-batched-tokens "$MAX_BATCHED" --block-size 128 \
    --engram-config '{"cpu_offload": false}' \
    --default-chat-template-kwargs "{\"thinking\": $THINKING}" \
    $TEXT_ARGS $PARSER_ARGS $SPEC_ARGS "${GRAPH_ARGS[@]}" \
    --distributed-executor-backend mp --nnodes 4 --node-rank "$NODE_RANK" \
    --master-addr "$HEAD_IP" --master-port "$MPORT" $HEADLESS $VLLM_EXTRA

# --- engine log capture on the HOST (a reboot or a later docker rm must never lose the only log) ---
mkdir -p ${DSV41_HOME:-$HOME/dsv41}/logs
setsid nohup docker logs -f "$NAME" > "${DSV41_HOME:-$HOME/dsv41}/logs/${NAME}-rank${NODE_RANK}-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 < /dev/null &
echo "  engine log -> ${DSV41_HOME:-$HOME/dsv41}/logs/${NAME}-rank${NODE_RANK}-*.log"

# --- load-time memory guard (stops the host OOM->reboot during weight load) ---
if sudo -n true 2>/dev/null; then
  DROP='sudo -n sh -c "echo 1 > /proc/sys/vm/drop_caches"'
else
  DROP='docker run --rm --privileged --entrypoint sh '"$IMAGE"' -c "echo 1 > /proc/sys/vm/drop_caches"'
fi
setsid nohup bash -c '
  for i in $(seq 1 300); do
    docker ps -q -f name='"$NAME"' | grep -q . || exit 0            # container gone: stop
    curl -sf -m 4 http://127.0.0.1:'"$PORT"'/v1/models >/dev/null 2>&1 && exit 0   # serving: stop
    a=$(( $(grep MemAvailable /proc/meminfo | awk "{print \$2}") / 1048576 ))
    if [ "$a" -lt 2 ]; then low=$((${low:-0}+1)); else low=0; fi
    if [ "${low:-0}" -ge 2 ]; then docker kill '"$NAME"' >/dev/null 2>&1; echo "$(date -u +%FT%TZ) MEMGUARD KILLED '"$NAME"' at MemAvailable=${a}GiB" >> ${DSV41_HOME:-$HOME/dsv41}/logs/memguard-rank'"$NODE_RANK"'.log; exit 0; fi
    [ "$a" -lt 6 ] && { '"$DROP"' >/dev/null 2>&1; }
    sleep 2
  done
' >/dev/null 2>&1 < /dev/null &
echo "  mem-guard armed: drop cache <6GiB, docker kill <2GiB for 2 consecutive samples -- steady state at gmu 0.78-0.80 is ~4-7GiB free (contained failure beats the hang-guard panic)"

echo "launched $NAME rank=$NODE_RANK extra=${EXTRA_PATCHES:-none} log=${LOG_LEVEL:-INFO} alloc=${ALLOC_CONF:-expandable_segments:True} cache=${CACHE_TAG:-shared} nccl=${NCCL_ENV_MODE:-legacy}${NCCL_DROP:+-drop:$NCCL_DROP}${NCCL_SET:+-set:$NCCL_SET} image=$IMAGE patches=$PATCH_DIR gmu=$GMU maxlen=$MAXLEN seqs=$SEQS eager=$EAGER spec=$SPEC text_only=$TEXT_ONLY avail=${AVAIL_GB}GiB"
sleep 3
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited" >&2; docker logs --tail 40 "$NAME" >&2; exit 1; }
