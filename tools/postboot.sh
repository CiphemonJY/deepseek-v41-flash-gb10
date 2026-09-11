#!/bin/bash
# postboot.sh <UTC-start> : run on the laptop after a test boot is READY. Battery on the head, then head-log + Xid evidence.
S="$1"; ssh -o ConnectTimeout=30 ${RANK0_SSH:?} 'export PYTHONIOENCODING=utf-8; python3 -u ${DSV41_HOME:-$HOME/dsv41}/battery.py' 2>&1 | tail -8
echo "=== head log signals ==="
ssh -o ConnectTimeout=30 ${RANK0_SSH:?} 'L=$(ls -t ${DSV41_HOME:-$HOME/dsv41}/logs/vllm_dsv41-rank0-*.log | head -1); echo "log: $L"
echo "  address-check assertions: $(grep -c "addresses changed between capture and replay" "$L")"; grep -m2 "addresses changed" "$L" | sed "s/^([A-Za-z_0-9 =]*) //" | cut -c1-200
st=$(grep -n "Application startup complete" "$L" | head -1 | cut -d: -f1); echo "  startup line: ${st:-none}"
[ -n "$st" ] && echo "  captures AFTER startup: $(tail -n +$st "$L" | grep -c "Captured breakable cudagraph")" && tail -n +$st "$L" | grep "Captured breakable cudagraph" | head -3 | sed "s/^([A-Za-z_0-9 =]*) //" | cut -c1-160
echo "  errors: $(grep -ciE "Error|Traceback" "$L")"; grep -iE "Error" "$L" | grep -vE "ERROR_HANDLING|error_file|duplicates" | head -2 | cut -c1-160'
echo "=== Xid since $S (the non-panicking rank first) ==="
for h in ${RANK1_SSH:?} ${RANK0_SSH:?} ${RANK2_SSH:?} ${RANK3_SSH:?}; do timeout 60 ssh -o ConnectTimeout=30 $h "bash ${DSV41_HOME:-$HOME/dsv41}/xid_since.sh $S" 2>&1 | head -4; done
