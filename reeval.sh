#!/usr/bin/env bash
# Re-score an existing checkpoint under a different test-time setting.
# No training -- a couple of minutes, not an hour and a half.
#
#   bash reeval.sh <run-name> [KEY VALUE]...
#
#   bash reeval.sh cargo_base TEST.NECK_FEAT after
#   bash reeval.sh cargo_base DATASETS.PROTOCOL AG
#   bash reeval.sh whu_pe_indep                       # view matrix, no retraining
#
# Three things this is for:
#
#   * NECK_FEAT.  'before' retrieves with the raw backbone feature, 'after'
#     with the BNNeck output.  HiHR does not say which it uses, and the choice
#     is worth a few tenths of mAP -- so it is worth one re-evaluation rather
#     than one retraining.
#   * DATASETS.PROTOCOL.  Note this only changes the *evaluation* split.  The
#     paper's A->A and G->G rows also retrain on the filtered data, so scoring
#     an ALL-trained checkpoint under AA is NOT the paper's A->A number.  AG is
#     the one protocol whose training set is the full one, so it is the only
#     sub-protocol this script can reproduce honestly.
#   * The view matrix.  Every WHU-MARS run so far was scored by modality only,
#     so aerial->ground retrieval -- the thing the positional correction exists
#     for -- has never actually been measured.  It needs no retraining: the
#     camera ids are already in the evaluator for the junk rule.
#
# The output is APPENDED to $REPO/reeval_log.txt as well as printed, because on
# this platform a task that succeeds keeps no log at all -- only a failed one
# can be downloaded afterwards.  Appending rather than truncating is what lets
# several re-evaluations share one task, which is how they are actually run:
#
#   for m in a b c; do bash reeval.sh $m TEST.NECK_FEAT after; done
#
# Clear it with `: > reeval_log.txt` when it gets long.  Override the path with
# REEVAL_LOG=/some/where.
set -euo pipefail

MODE=${1:-}
[ -n "$MODE" ] || { echo "usage: bash reeval.sh <run-name> [KEY VALUE]..." >&2; exit 1; }
shift

PY=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
REPO=/mnt/cache/wanghanzhi/HSY/HiHR
DATA=/mnt/cache/wanghanzhi/Datasets

CFG=$REPO/configs/hihr_$MODE.yml
# Checkpoint name follows SOLVER.MAX_EPOCHS, and the initialisation follows the
# config's own PRETRAIN_PATH.  Both were hardcoded to 60 epochs and to
# cargo_base, which silently excluded the 120-epoch runs and would have handed
# whu_twin the wrong starting weights.
EPOCHS=$(grep -aE '^ *MAX_EPOCHS:' "$CFG" | head -1 | tr -dc '0-9')
CKPT=$REPO/out_hihr_$MODE/transformer_${EPOCHS:-60}.pth

[ -f "$CFG" ]  || { echo "missing config: $CFG" >&2; exit 1; }
[ -f "$CKPT" ] || { echo "missing checkpoint: $CKPT" >&2; exit 1; }

cd "$REPO"

EXTRA=()

# PRETRAIN_PATH still has to be loadable even though TEST.WEIGHT overwrites
# everything a moment later: make_model reads it during construction.  A
# PRETRAIN_CHOICE of 'self' expects a checkpoint from this codebase -- handing
# it the raw CLIP archive makes load_param raise -- and which checkpoint that
# is comes from the config, not from the run's name.
if grep -q "^ *PRETRAIN_CHOICE: *'self'" "$CFG"; then
  SELF=$(grep -aE "^ *PRETRAIN_PATH:" "$CFG" | head -1 | sed "s/.*'\([^']*\)'.*/\1/")
  EXTRA+=(MODEL.PRETRAIN_PATH "$REPO/${SELF#./}")
else
  EXTRA+=(MODEL.PRETRAIN_PATH "$DATA/ViT-B-16.pt")
fi

# Same content-based check as run_hihr.sh: a name glob would miss whu_mtext_*.
if grep -q '^ *TEXT_ALIGN: *True' "$CFG"; then
  EXTRA+=(MODEL.TEXT_CLIP_PATH "$DATA/ViT-B-16.pt")
fi

# Every grep that probes a config is anchored to the start of a line, so that a
# key NAMED IN A COMMENT cannot flip a branch.  hihr_whu_ce_mod_g2.yml says
# "AERIAL_CAMS stays empty" in its header; an unanchored grep read that as the
# key being set and silently skipped the view matrix for that one run.
# The view matrix is keyed off DATASETS.AERIAL_CAMS, which the runs made before
# direction two do not set.  Injecting it here is what lets those checkpoints be
# compared against the text-supervised ones on the same readout.  c6/c7 -> zero
# based [5, 6], confirmed by crop geometry, and WHU-MARS only.
if grep -q "^ *NAMES: *('WHU-MARS')" "$CFG" &&
   ! grep -q '^ *AERIAL_CAMS:' "$CFG"; then
  EXTRA+=(DATASETS.AERIAL_CAMS "[5, 6]")
fi

LOG=${REEVAL_LOG:-$REPO/reeval_log.txt}

# The overrides are echoed rather than left to be inferred from the numbers.
# This script exists to compare settings, so a block of results that does not
# name its own setting is the one way its output can be actively misleading --
# and two runs of the same checkpoint differing only in NECK_FEAT look
# identical apart from the digits.
{
  echo
  echo "=== $(date '+%F %T')  reeval $MODE"
  echo "=== checkpoint : $CKPT"
  echo "=== overrides  : ${*:-<none, config defaults>}"
} | tee -a "$LOG"

# The filter below keeps `loaded (split:` -- the dataset banner.  Without it the
# log records which override was REQUESTED but never what the run actually read,
# and a split or protocol that failed to take effect would look exactly like one
# that worked.  The numbers would simply be quietly wrong, under the right name.
"$PY" -u test.py --config_file "$CFG" \
  DATASETS.ROOT_DIR "$DATA" \
  TEST.WEIGHT "$CKPT" \
  OUTPUT_DIR /tmp/reeval_$MODE \
  "${EXTRA[@]}" "$@" 2>&1 \
  | grep -aE 'mAP|Rank-|Pair|View|Loading pretrained|loaded [0-9]+|loaded +\(split|NECK_FEAT|PROTOCOL|feature is (NOT )?normalized' \
  | tee -a "$LOG"
