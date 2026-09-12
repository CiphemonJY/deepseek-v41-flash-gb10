#!/bin/bash
# Post-reboot readiness report for one rank of a TP serve. No GPU allocation, no state change.
# Set FABRIC_IFACE_RE to match your fabric NICs (default: any en*/ib* interface).
# Set PATCH_DIR / CKPT_DIR if yours are not $HOME/patches/dsv41-boot10 and $HOME/dsv41/hf-ckpt.
echo "host=$(hostname -s) up=$(uptime -p | sed 's/^up //')"
echo "dockerd=$(systemctl is-active docker 2>/dev/null)"
echo "ctrs=[$(docker ps -a --format '{{.Names}}:{{.Status}}' 2>/dev/null | tr '\n' ',')]"
echo "mem_avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))GiB"
echo "ib_dev=[$(ls /dev/infiniband 2>/dev/null | tr '\n' ' ')]"
echo "roce_sysfs=[$(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')]"
ip -br addr show 2>/dev/null | grep -E "${FABRIC_IFACE_RE:-^(en|ib)}" | sed 's/^/nic: /'
echo "gpu=[$(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>&1 | head -1)]"
echo "xid_since_boot=$(dmesg 2>/dev/null | grep -c 'Xid' || echo na)"
echo "images_vllm_dsv41=$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -c '^vllm-dsv41:')"
echo "image_tags=[$(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep '^vllm-dsv41:' | tr '\n' ' ')]"
echo "extra_mounts=$(test -f "${PATCH_DIR:-$HOME/patches/dsv41-boot10}/mounts.extra.txt" && echo PRESENT || echo absent)"
echo -n "patch_md5="
for f in engram.py model_state.py attention.py flashinfer_sparse.py sparse_swa.py sparse_attn_indexer.py weight_utils.py mounts.txt; do
  p="${PATCH_DIR:-$HOME/patches/dsv41-boot10}/$f"
  if [ -f "$p" ]; then printf '%s:%s ' "$f" "$(md5sum "$p" | cut -c1-8)"; else printf '%s:MISSING ' "$f"; fi
done; echo
echo -n "patch_cr_bytes="
tot=0; for f in engram.py model_state.py attention.py flashinfer_sparse.py sparse_swa.py sparse_attn_indexer.py weight_utils.py mounts.txt; do
  p="${PATCH_DIR:-$HOME/patches/dsv41-boot10}/$f"; [ -f "$p" ] && tot=$(( tot + $(tr -cd '\r' < "$p" | wc -c) )); done; echo "$tot"
echo "ckpt=$(test -f "${CKPT_DIR:-$HOME/dsv41/hf-ckpt}/model-00048-of-00048.safetensors" && echo complete || echo INCOMPLETE)"
echo "pause_markers=[$(ls "${WATCH_DIR:-$HOME/ds4}" 2>/dev/null | grep -i paused | tr '\n' ' ')]"
echo "hang_guard=[$(cat "${HANG_GUARD_CONF:-/etc/sysctl.d/90-oom-hang-guard.conf}" 2>/dev/null | grep -vE '^#|^$' | tr '\n' ' ')]"
