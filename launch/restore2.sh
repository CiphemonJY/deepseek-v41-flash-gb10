#!/bin/bash
# Clean restore of a TP serve after a failed/aborted boot lost its rendezvous (workers hit
# "Gloo connectFullMesh timed out" because the previous experiment's containers were still tearing down).
# Order: pause watchdog -> HARD stop every rank and PROVE none are left -> settle -> fresh boot -> verify ->
# re-arm the watchdog ONLY if the serve is verified healthy.
set -uo pipefail
RANKS="${RANK0_SSH:?} ${RANK2_SSH:?} ${RANK3_SSH:?}"
HOP="${RANK2_SSH:?}"          # a reachable worker used to hop to rank1
RANK1="${RANK1_SSH:?}"        # rank1 ssh target reachable FROM $HOP
P="${PATCH_DIR:-$HOME/patches/dsv41-boot10}"
MODEF="${STAGE_MODE_FILE:-$HOME/dsv41/vllm-cache/dsv41_stage_mode}"

echo "=== 1. pause the watchdog on the head (it must not race this restore) ==="
timeout 40 ssh -o ConnectTimeout=30 ${RANK0_SSH} 'touch ~/ds4/DSV41_WATCH_PAUSED; ls ~/ds4 | grep -i paused | tr "\n" " "; echo'

echo "=== 2. hard stop every rank + clear experiment residue ==="
for h in $RANKS; do
  timeout 90 ssh -o ConnectTimeout=30 $h "docker rm -f vllm_dsv41 >/dev/null 2>&1; rm -f $P/mounts.extra.txt $MODEF; printf '  %-11s left=[%s]\n' \"\$(hostname -s)\" \"\$(docker ps -a --format '{{.Names}}' -f name=vllm_dsv41 | tr '\n' ' ')\""
done
timeout 150 ssh -o ConnectTimeout=30 $HOP "timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 $RANK1 \"docker rm -f vllm_dsv41 >/dev/null 2>&1; rm -f $P/mounts.extra.txt $MODEF; printf '  %-11s left=[%s]\n' \\\"\\\$(hostname -s)\\\" \\\"\\\$(docker ps -a --format '{{.Names}}' -f name=vllm_dsv41 | tr '\n' ' ')\\\"\""

echo "=== 3. settle (let the fabric sockets close before a new rendezvous) ==="
for i in $(seq 1 6); do sleep 10; done
for h in $RANKS; do timeout 40 ssh -o ConnectTimeout=30 $h 'printf "  %-11s mem=%sGiB listeners_on_29500=%s\n" "$(hostname -s)" "$(( $(awk "/MemAvailable/{print \$2}" /proc/meminfo) / 1048576 ))" "$(ss -tlnH 2>/dev/null | grep -c 29500)"'; done

echo "=== 4. fresh boot from the head (verified production knobs) ==="
timeout 1500 ssh -o ConnectTimeout=30 ${RANK0_SSH} 'bash $HOME/dsv41/restore_prod.sh' 2>&1 \
  | grep -E "cleaned|launched vllm_dsv41|GATE FAIL|Refusing|boot exit|FAILED" | cut -c1-140

echo "=== 5. wait for :8888 ==="
READY=no
for i in $(seq 1 60); do
  R=$(timeout 30 ssh -o ConnectTimeout=20 ${RANK0_SSH} 'up=$(docker ps -q -f name=vllm_dsv41 | head -1); c=$(curl -s -o /dev/null -w "%{http_code}" -m 4 http://127.0.0.1:8888/v1/models); printf "%s :8888=%s" "${up:+up}" "$c"' 2>/dev/null)
  case "$R" in
    *":8888=200") echo "  [$(date -u +%H:%M:%SZ)] READY"; READY=yes; break ;;
    " :8888"*) echo "  >>> CONTAINER EXITED"; timeout 60 ssh ${RANK0_SSH} 'L=$(ls -t $HOME/dsv41/logs/vllm_dsv41-rank0-*.log | head -1); grep -E "DistStoreError|connectFullMesh|Error" "$L" | grep -vE "ERROR_HANDLING|error_file" | tail -4 | cut -c1-190'; break ;;
  esac
  sleep 20
done

if [ "$READY" != yes ]; then
  echo ">>> RESTORE FAILED. Watchdog left PAUSED deliberately (it would relaunch into the same failure)."
  echo ">>> Ranks:"; for h in $RANKS; do timeout 40 ssh -o ConnectTimeout=30 $h 'printf "  %-11s ctr=[%s]\n" "$(hostname -s)" "$(docker ps -a --format "{{.Status}}" -f name=vllm_dsv41 | head -1)"'; done
  exit 1
fi

echo "=== 6. verify ==="
timeout 60 ssh ${RANK0_SSH} 'curl -s -m 10 http://127.0.0.1:8888/v1/models | python3 -c "import json,sys; print(\"  served:\", [m[\"id\"] for m in json.load(sys.stdin)[\"data\"]])"; L=$(ls -t $HOME/dsv41/logs/vllm_dsv41-rank0-*.log | head -1); echo "  error lines: $(grep -cE "ERROR|Traceback" "$L")"'
timeout 900 ssh ${RANK0_SSH} 'export PYTHONIOENCODING=utf-8; cd "${REMOTE_HOME_DSV41:-$HOME/dsv41}" && python3 garble_gate.py compare http://127.0.0.1:8888/v1 2>&1 | tail -1 | sed "s/^/  /"'
GATE=$(timeout 900 ssh ${RANK0_SSH} 'export PYTHONIOENCODING=utf-8; cd "${REMOTE_HOME_DSV41:-$HOME/dsv41}" && python3 garble_gate.py compare http://127.0.0.1:8888/v1 2>&1 | tail -1' 2>/dev/null)
timeout 900 ssh ${RANK0_SSH} 'export PYTHONIOENCODING=utf-8; cd "${REMOTE_HOME_DSV41:-$HOME/dsv41}" && timeout 800 python3 -u vision_tools_demo.py http://127.0.0.1:8888/v1 2>&1 | grep -E "summary" | sed "s/^/  /"'
echo "=== 7. Xid ==="
for h in $RANKS; do timeout 60 ssh -o ConnectTimeout=30 $h "bash $HOME/dsv41/xid_since.sh $(date -u -d '20 minutes ago' +%Y-%m-%dT%H:%M:%SZ)" 2>&1 | head -1 | sed 's/^/  /'; done

case "$GATE" in
  *PASS*)
    echo "=== 8. re-arm the watchdog (gate PASSED) ==="
    for h in $RANKS; do timeout 40 ssh -o ConnectTimeout=30 $h 'rm -f ~/ds4/DSV41_WATCH_PAUSED; printf "  %-11s markers=[%s]\n" "$(hostname -s)" "$(ls ~/ds4 | grep -i paused | tr "\n" " ")"'; done
    timeout 180 ssh ${RANK0_SSH} 'id1=$(docker inspect -f "{{.Id}}" vllm_dsv41); bash $HOME/dsv41/dsv41_watch.sh; sleep 2; id2=$(docker inspect -f "{{.Id}}" vllm_dsv41); echo "  watch log: $(tail -1 ~/ds4/dsv41_watch.log)"; [ "$id1" = "$id2" ] && echo "  container UNCHANGED (no-op on a healthy serve)" || echo "  !!! container CHANGED"'
    ;;
  *) echo "=== 8. gate did NOT pass ($GATE) -- watchdog left PAUSED for review ===" ;;
esac
echo "RESTORE2 DONE $(date -u +%FT%TZ)"
