#!/usr/bin/env bash
# boot41.sh [start|stop] -- worker-first TP4 boot of DeepSeek-V4.1-Flash on this fleet, or stop it.
# Runs from the laptop over the ssh aliases. Rank map via RANK{0..3}_SSH.
# Mirrors tonyd2wild's boot_dsv41.sh: ranks 3,2,1 first with IDENTICAL knobs, then the head.
# STOP is head-first: a worker that starts while an old head still listens joins its rendezvous and hangs.
set -uo pipefail
MODE="${1:-start}"
declare -A HOST=( [0]="${RANK0_SSH:?ssh alias/user@host for rank 0}" [1]="${RANK1_SSH:?}" [2]="${RANK2_SSH:?}" [3]="${RANK3_SSH:?}" )
KNOBS=""
for k in NCCL_DROP NCCL_SET NCCL_ENV_MODE IMAGE GMU MAXLEN SEQS MAX_BATCHED EAGER CUDAGRAPH_MODE CG_SIZES SPEC SPEC_K SPEC_ADAPT TEXT_ONLY THINKING PARSERS PATCH_NAME VLLM_EXTRA; do
  v="${!k:-}"; [ -n "$v" ] && KNOBS="$KNOBS $k='$v'"
done
S='ssh -o ConnectTimeout=25 -o BatchMode=yes'
if [ "$MODE" = "stop" ]; then
  for r in 0 3 2 1; do printf "stop rank %s (%s): " "$r" "${HOST[$r]}"; timeout 60 $S "${HOST[$r]}" 'docker rm -f vllm_dsv41 >/dev/null 2>&1 && echo stopped || echo none' 2>&1 | tail -1; done
  exit 0
fi
echo "knobs:$KNOBS"
# council (systems seat): stop ALL ranks head-first before starting, so no worker joins a stale head's rendezvous
for r in 0 3 2 1; do timeout 60 $S "${HOST[$r]}" 'docker rm -f vllm_dsv41 >/dev/null 2>&1; true' 2>/dev/null; done
for r in 3 2 1; do
  echo "== rank $r (${HOST[$r]}) =="
  timeout 240 $S "${HOST[$r]}" "export $KNOBS; bash ${DSV41_HOME:-$HOME/dsv41}/launch41.sh $r" 2>&1 | tail -3 || { echo "rank $r launch FAILED" >&2; exit 1; }
done
sleep 5
echo "== rank 0 (${HOST[0]}, head) =="
timeout 240 $S "${HOST[0]}" "export $KNOBS; bash ${DSV41_HOME:-$HOME/dsv41}/launch41.sh 0" 2>&1 | tail -3 || { echo "head launch FAILED" >&2; exit 1; }
echo "all four launched $(date -u +%FT%TZ)"
