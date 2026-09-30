#!/usr/bin/env bash
# Run several run_hihr.sh modes back to back inside ONE task.
#
#   bash run_seq.sh whu_pe_indep_clip whu_pe_chain_clip whu_mcam_clip
#
# The platform takes one launch command per task and one task holds a machine,
# so five experiments normally means five machines.  This is the same five
# experiments on one, at the cost of running them in series.
#
# Three things it does that a bare `for m in ...; do bash run_hihr.sh $m; done`
# would not, each of which is worth its lines over a ten-hour task:
#
#   * PRE-FLIGHT.  Every config and every checkpoint it names is checked before
#     the first run starts.  A typo in the fifth mode otherwise surfaces eight
#     hours in, with the machine already spent.
#   * RESUME.  A mode whose final checkpoint is already on the shared drive is
#     skipped.  If the task is reclaimed at run four, relaunching the same
#     command continues from run four instead of redoing three.
#   * ONE FAILURE DOES NOT COST THE REST.  run_hihr.sh exits non-zero when a
#     copy-back fails; under `set -e` that would abandon the remaining runs
#     even though the machine is fine.  Failures are recorded and reported at
#     the end, and the exit status still reflects them.
#
# Progress goes to $REPO/run_seq_log.txt as well as stdout, because a task that
# SUCCEEDS keeps no downloadable log on this platform.
set -uo pipefail          # deliberately not -e; see above

MODES=("$@")
[ ${#MODES[@]} -gt 0 ] || { echo "usage: bash run_seq.sh <mode> [mode ...]" >&2; exit 1; }

REPO=/mnt/cache/wanghanzhi/HSY/HiHR
DATA=/mnt/cache/wanghanzhi/Datasets
LOG=${SEQ_LOG:-$REPO/run_seq_log.txt}
SKIP_DONE=${SKIP_DONE:-1}

cd "$REPO"

say() { echo "$@" | tee -a "$LOG"; }

# ---------------------------------------------------------------- pre-flight
say ""
say "=== $(date '+%F %T')  run_seq: ${MODES[*]}"
bad=0
for m in "${MODES[@]}"; do
  CFG=$REPO/configs/hihr_$m.yml
  if [ ! -f "$CFG" ]; then say "  MISSING CONFIG  $CFG"; bad=1; continue; fi
  # Same rule run_hihr.sh uses, and anchored the same way: a key named in a
  # COMMENT must not decide the branch.
  if grep -q "^ *PRETRAIN_CHOICE: *'self'" "$CFG"; then
    SELF=$(grep -aE "^ *PRETRAIN_PATH:" "$CFG" | head -1 | sed "s/.*'\([^']*\)'.*/\1/")
    W=$REPO/${SELF#./}
  else
    W=$DATA/ViT-B-16.pt
  fi
  [ -f "$W" ] || { say "  MISSING WEIGHTS $W  (needed by $m)"; bad=1; }
  if grep -q '^ *TEXT_ALIGN: *True' "$CFG" && [ ! -f "$DATA/ViT-B-16.pt" ]; then
    say "  MISSING CLIP    $DATA/ViT-B-16.pt  (needed by $m)"; bad=1
  fi
done
[ "$bad" = 0 ] || { say "pre-flight failed; nothing was started."; exit 1; }
say "  pre-flight ok: ${#MODES[@]} runs queued"

# --------------------------------------------------------------------- loop
started=$(date +%s)
failed=()
skipped=()
for m in "${MODES[@]}"; do
  CFG=$REPO/configs/hihr_$m.yml
  # Checkpoint name follows SOLVER.MAX_EPOCHS, exactly as run_hihr.sh and
  # reeval.sh derive it, so a 90-epoch mode is not mistaken for unfinished.
  EP=$(grep -aE '^ *MAX_EPOCHS:' "$CFG" | head -1 | tr -dc '0-9')
  DONE=$REPO/out_hihr_$m/transformer_${EP:-60}.pth
  if [ "$SKIP_DONE" = 1 ] && [ -f "$DONE" ]; then
    say ""
    say "--- $(date '+%F %T')  SKIP $m (already at $DONE)"
    skipped+=("$m")
    continue
  fi

  say ""
  say "--- $(date '+%F %T')  START $m   ($((1 + ${#failed[@]} + ${#skipped[@]}))/${#MODES[@]} through the queue)"
  t0=$(date +%s)
  bash run_hihr.sh "$m"
  rc=$?
  t1=$(date +%s)
  if [ "$rc" = 0 ]; then
    say "--- $(date '+%F %T')  DONE  $m in $(( (t1-t0)/60 )) min"
  else
    say "--- $(date '+%F %T')  FAILED $m (exit $rc) after $(( (t1-t0)/60 )) min"
    say "    its own log is $REPO/train_log_hihr_$m.txt; the queue continues"
    failed+=("$m")
  fi
done

# ------------------------------------------------------------------ summary
say ""
say "=== $(date '+%F %T')  run_seq finished in $(( ($(date +%s)-started)/60 )) min"
[ ${#skipped[@]} -eq 0 ] || say "    skipped (already done): ${skipped[*]}"
if [ ${#failed[@]} -eq 0 ]; then
  say "    all runs completed"
else
  say "    FAILED: ${failed[*]}"
  say "    relaunch the same command -- finished runs are skipped, these are retried"
  exit 1
fi
