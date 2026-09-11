#!/bin/bash
# xid_since.sh <UTC ISO start, e.g. 2026-09-11T16:05:00Z> -- Xid/NVRM/NCCL-relevant kernel lines on this rank since then (journal is local time)
S=$(date -d "$1" "+%Y-%m-%d %H:%M:%S"); H=$(hostname)
L=$(journalctl -k --since "$S" --no-pager 2>/dev/null | grep -iE "Xid|NVRM|nvidia-uvm|hung task|soft lockup|NMI" | grep -vE "NV_ERR_NO_MEMORY")
n=$(printf "%s" "$L" | grep -c .); echo "$H: $n kernel fault lines since $1"; printf "%s\n" "$L" | sed -E 's/^[A-Za-z]{3} [0-9]{2} //' | cut -c1-140 | head -8
