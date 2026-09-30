#!/usr/bin/env bash
# Two legs, run in this order:
#
#   bash run_hihr.sh cargo_base       # 1. reproduce HiHR's baseline on CARGO
#   bash run_hihr.sh whu_from_cargo   # 2. WHU-MARS, starting from that checkpoint
#
# Leg 1 is HiHR arXiv:2607.09186v1 sections 4.2 / 4.4: CLIP-ViT-B/16 +
# cross-entropy + triplet, Adam, 60 epochs, two-tier learning rate (5e-6
# pretrained / 3.5e-4 fresh), 100-iteration warmup then cosine.  Target is
# table 4's Baseline row, 63.61 mAP / 68.23 Rank-1.
#
# Leg 2 carries leg 1's weights onto WHU-MARS.  Its result is the new baseline;
# nothing from the earlier ViT-B line is a comparison point for it.
set -euo pipefail

MODE=${1:-}
case "$MODE" in
  cargo_base)         ;;  # leg 1: reproduce HiHR's baseline on CARGO
  whu_from_cargo)     ;;  # leg 2: WHU-MARS from that checkpoint -> mAP 10.02 / R-1 25.32
  whu_from_cargo_120) ;;  # ... same, 120 epochs. Only MAX_EPOCHS differs.
  whu_recipe_hihr)    ;;  # HiHR's recipe verbatim -> mAP 10.23 / R-1 26.68  (the baseline)
  whu_recipe_mars)    ;;  # WHU-MARS's recipe verbatim (cancelled 2026-08-05)
  whu_recipe_clip)    ;;  # ... unsplit CE but straight from raw CLIP -- the 2x2's fourth cell
  whu_pe_indep_clip)  ;;  # the five earlier ideas, re-taken on the raw CLIP tower:
  whu_rope_indep_clip) ;;  # rotary as the residual: same skeleton as pe_indep, alpha=0 init
  whu_rope_full_clip) ;;   # ... its control: rotation at full strength from step 0
  whu_rope_freq_clip) ;;   # ... alpha=0 plus trainable frequencies
  whu_rope_chart_clip) ;;  # ... TPCG per-image coordinates instead of the fixed grid
  whu_rope_chart_freq_clip) ;; # ... TPCG coordinates AND trainable frequencies (the 2x2 corner)
  whu2337_text_lam50) ;;      # the larger split: main method, 2 anchors
  whu2337_mtext_lam50) ;;     # ...              6 anchors (best mAP on the 1000 split)
  whu2337_ce_mod_g2_mtext) ;; # ...              everything stacked (best R-1 there)
  whu_pe_chain_clip)  ;;  # ... direction 1, both halves of its ablation
  whu_text_lam50_clip) ;; # ... direction 2, viewpoint axis, 2 anchors
  whu_mtext_lam50_clip) ;; # ...                             6 anchors
  whu_mcam_clip)      ;;  # ... direction 2, SPECTRUM axis -- the one whose cause of death moved
  whu_mcam_grp_clip)  ;;  # ... same but only Thermal is corrected -- worth +0.94 on the g2 tower
  whu_ce_mod_g2_pe_clip) ;;   # stacking: CE grouping + positional delta
  whu_ce_mod_g2_mtext_clip) ;; # ...      + the viewpoint-axis text loss on top
  whu_ce_mod_g2_e90_clip) ;;  # ... 90 epochs: the raw tower has had 60 fewer than the cargo one
  whu_ce_mod_clip)    ;;  # three-way spectrum split on the raw tower -- the grouping choice, re-measured
  whu_text_lam10_clip) ;; # text weight sweep on the raw tower, 2 anchors: lambda 1.0
  whu_text_lam100_clip) ;; # ...                                              lambda 10.0
  whu_recipe_clip_e90) ;;      # the BASELINE at 90 epochs -- every increment is measured on it
  whu_text_lam50_e90_clip) ;;  # ... and the main method at 90
  whu_text_lam50_pedlr05_clip) ;; # the delta learning-rate multiplier: 0.5
  whu_text_lam50_pedlr20_clip) ;; # ...                                  2.0
  cargo_pe_text)      ;;  # does the method transfer? delta + viewpoint text, on CARGO
  cargo_pe)           ;;  # ... its control: delta only, text weight 0
  whu_pe_indep)       ;;  # direction 1: layer-wise residuals, independent -> 10.36
  whu_pe_chain)       ;;  # direction 1: ... carried forward -> 10.30
  whu_text_lam01)     ;;  # direction 2: + CLIP text supervision, lambda 0.1
  whu_text_lam10)     ;;  # ...                                  lambda 1.0
  whu_text_lam50)     ;;  # ...                                  lambda 5.0 -> 10.45
  whu_mtext_lam50)    ;;  # ... six anchors (modality slot),     lambda 5.0
  whu_mtext_lam100)   ;;  # ...                                  lambda 10.0
  whu_mcam)           ;;  # text_lam50 with the axis swapped to the SPECTRUM -> 3 anchors
  whu_tnce)           ;;  # twin contrastive: per-image target, trainable backbone
  whu_tnce_warm)      ;;  # ... weight 1.0 ramped in over 10 epochs
  whu_tnce_lam25)     ;;  # ... weight 0.25, flat
  whu_tnce_ctrl)      ;;  # ... weight 0, the blank control for the pipeline
  whu_tnce_lam25_re)  ;;  # ... weight 0.25 with erasing untied, not removed
  whu_tnce_ctrl_re)   ;;  # ... weight 0 with erasing untied -- the 2x2's fourth corner
  whu_tnce_lam50_re)  ;;  # ... lambda sweep on the untied arm: 0.5
  whu_tnce_lam100_re) ;;  # ...                                 1.0
  whu_tnce_lam200_re) ;;  # ...                                 2.0
  whu_ce_view)        ;;  # split the CE label by viewpoint (500 -> 1000)
  whu_ce_mod)         ;;  # ... by spectrum            (500 -> 1500) -> 10.49 / 34.77
  whu_ce_mod_g2)      ;;  # ... Thermal alone, RGB+IR merged  (500 -> 1000) -> 11.58 / 32.71
  whu_ce_mod_g2_e40)  ;;  # ... g2 on a 40-epoch cosine schedule
  whu_ce_mod_g2_e50)  ;;  # ...             50-epoch
  whu_ce_mod_g2_e90)  ;;  # ...             90-epoch -> 11.26 / 30.37
  whu_ce_mod_g2_e120) ;;  # ...            120-epoch -> 10.05 / 26.12
  whu_ce_mod_g2_re70) ;;  # ... g2 with random erasing at 0.7 instead of 0.5 -> 11.60 (null)
  whu_ce_mod_g2_tri2) ;;  # ... g2 with the triplet weighted 2.0 -> 11.74 / 32.15
  whu_ce_mod_g2_tri23) ;; # ...                              2.3
  whu_ce_mod_g2_tri26) ;; # ...                              2.6
  whu_ce_mod_g2_tri3) ;;  # ...                              3.0 -> 11.70 / 31.46
  whu_ce_mod_g2_tri4) ;;  # ...                              4.0 -> 11.58 / 30.69
  whu_ce_mod_g2_regall) ;; # regularisation: all four at once -- read this one first
  whu_ce_mod_g2_dp20) ;;  # ... stochastic depth 0.1 -> 0.2
  whu_ce_mod_g2_dp30) ;;  # ...                         0.3
  whu_ce_mod_g2_attn10) ;; # ... attention dropout 0.0 -> 0.1
  whu_ce_mod_g2_wd5e4) ;; # ... weight decay 1e-4 -> 5e-4 (repo default)
  whu_ce_mod_g2_ls) ;;    # ... label smoothing off -> on  (repo default)
  whu_ce_mod_g2_ls_tri23) ;; # ... label smoothing + triplet 2.3, stacked
  whu_ce_mod_g2_clip) ;;  # ... straight from the raw CLIP tower, no CARGO leg
  whu_ce_mod_g2_pe) ;;    # text line: positional delta only, text weight 0 (control)
  whu_ce_mod_g2_mcam) ;;  # ... + spectrum-axis text correction, all -> RGB
  whu_ce_mod_g2_mcam_grp) ;; # ... RGB+IR left alone, only Thermal corrected
  whu_ce_both)        ;;  # ... by both                (500 -> 3000)
  whu_sync)           ;;  # HiHR recipe, one key changed: frame-synchronised triplets
  whu_twin)           ;;  # frozen tower, only the per-modality banks; twin-anchored
  *) echo "usage: bash run_hihr.sh {cargo_base|whu_from_cargo|whu_from_cargo_120|" >&2
     echo "                         whu_recipe_hihr|whu_recipe_mars|whu_pe_indep|whu_pe_chain|" >&2
     echo "                         whu_text_lam01|whu_text_lam10|whu_text_lam50|" >&2
     echo "                         whu_mtext_lam50|whu_mtext_lam100|whu_mcam|" >&2
     echo "                         whu_sync|whu_twin|whu_tnce|" >&2
     echo "                         whu_tnce_warm|whu_tnce_lam25|whu_tnce_ctrl|" >&2
     echo "                         whu_tnce_lam25_re|whu_tnce_ctrl_re|" >&2
     echo "                         whu_tnce_lam50_re|whu_tnce_lam100_re|" >&2
     echo "                         whu_tnce_lam200_re|" >&2
     echo "                         whu_ce_view|whu_ce_mod|whu_ce_both|whu_ce_mod_g2|" >&2
     echo "                         whu_ce_mod_g2_e40|whu_ce_mod_g2_e50|" >&2
     echo "                         whu_ce_mod_g2_e90|whu_ce_mod_g2_e120|" >&2
     echo "                         whu_ce_mod_g2_re70|whu_ce_mod_g2_tri2|" >&2
     echo "                         whu_ce_mod_g2_tri23|whu_ce_mod_g2_tri26|" >&2
     echo "                         whu_ce_mod_g2_tri3|whu_ce_mod_g2_tri4|" >&2
     echo "                         whu_ce_mod_g2_regall|whu_ce_mod_g2_dp20|" >&2
     echo "                         whu_ce_mod_g2_dp30|whu_ce_mod_g2_attn10|" >&2
     echo "                         whu_ce_mod_g2_wd5e4|whu_ce_mod_g2_ls|" >&2
     echo "                         whu_ce_mod_g2_pe|whu_ce_mod_g2_mcam|" >&2
     echo "                         whu_ce_mod_g2_mcam_grp|whu_ce_mod_g2_ls_tri23|" >&2
     echo "                         whu_ce_mod_g2_clip|whu_recipe_clip|" >&2
     echo "                         whu_pe_indep_clip|whu_pe_chain_clip|" >&2
     echo "                         whu_text_lam50_clip|whu_mtext_lam50_clip|" >&2
     echo "                         whu_rope_indep_clip|whu_rope_full_clip|" >&2
     echo "                         whu_rope_freq_clip|whu_rope_chart_clip|" >&2
     echo "                         whu_rope_chart_freq_clip|" >&2
     echo "                         whu2337_text_lam50|whu2337_mtext_lam50|" >&2
     echo "                         whu2337_ce_mod_g2_mtext|" >&2
     echo "                         whu_mcam_clip|whu_mcam_grp_clip|" >&2
     echo "                         whu_ce_mod_g2_pe_clip|whu_ce_mod_g2_mtext_clip|" >&2
     echo "                         whu_ce_mod_g2_e90_clip|whu_ce_mod_clip|" >&2
     echo "                         whu_text_lam10_clip|whu_text_lam100_clip|" >&2
     echo "                         whu_recipe_clip_e90|whu_text_lam50_e90_clip|" >&2
     echo "                         whu_text_lam50_pedlr05_clip|whu_text_lam50_pedlr20_clip|" >&2
     echo "                         cargo_pe_text|cargo_pe}" >&2
     exit 1 ;;
esac

PY=/mnt/cache/wanghanzhi/envs/llmpar/bin/python3
REPO=/mnt/cache/wanghanzhi/DFEE/VPR-ReID
DATA=/mnt/cache/wanghanzhi/Datasets
WEIGHT=$DATA/ViT-B-16.pt          # sha256 5806e77c... verified official OpenAI release

CFG=$REPO/configs/hihr_$MODE.yml
OUT=/tmp/out_hihr_$MODE           # scratch on the worker; /tmp is wiped between allocations
LOG=$REPO/train_log_hihr_$MODE.txt

[ -f "$CFG" ]    || { echo "missing config: $CFG" >&2; exit 1; }

# Which checkpoint a run starts from is read from its own config rather than
# inferred from its name.  Most WHU-MARS runs inherit leg 1's CARGO weights, but
# whu_twin starts from the clean whu_recipe_hihr baseline -- a `whu_*` glob
# would have handed it the wrong initialisation without saying so.
if grep -q "^ *PRETRAIN_CHOICE: *'self'" "$CFG"; then
  SELF=$(grep -E "^ *PRETRAIN_PATH:" "$CFG" | head -1 | sed "s/.*'\([^']*\)'.*/\1/")
  WEIGHT=$REPO/${SELF#./}
  [ -f "$WEIGHT" ] || { echo "missing checkpoint named by $CFG:" >&2
                        echo "  $WEIGHT" >&2
                        echo "train that run first." >&2; exit 1; }
fi

[ -f "$WEIGHT" ] || { echo "missing weights: $WEIGHT" >&2; exit 1; }

cd "$REPO"

{
  echo "=== $(date '+%F %T')  hihr/$MODE"
  echo "=== config  : $CFG"
  echo "=== weights : $WEIGHT"
  echo "=== commit  : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
} > "$LOG"

EXTRA=()
# Keyed off what the config actually asks for, not off the branch name: a glob
# like whu_text_* silently misses whu_mtext_*, and the run would then fail deep
# inside model construction rather than here.
if grep -q '^ *TEXT_ALIGN: *True' "$CFG"; then
  # The text tower comes from the original CLIP release, not from PRETRAIN_PATH
  # -- that points at our CARGO checkpoint, which only ever held the image
  # tower, so loading a text tower from it would find nothing.
  [ -f "$DATA/ViT-B-16.pt" ] || { echo "missing CLIP release: $DATA/ViT-B-16.pt" >&2; exit 1; }
  EXTRA+=(MODEL.TEXT_CLIP_PATH "$DATA/ViT-B-16.pt")
fi

"$PY" -u train.py --config_file "$CFG" \
  MODEL.PRETRAIN_PATH "$WEIGHT" \
  DATASETS.ROOT_DIR "$DATA" \
  OUTPUT_DIR "$OUT" "${EXTRA[@]}" >> "$LOG" 2>&1

# Copy the checkpoint back, verified byte for byte and retried.
#
# On 2026-08-12 the shared drive filled up DURING a run -- three checkpoints
# copied back at the same moment and all three landed truncated (54M / 73M /
# 54M against ~338M).  Nothing said so at the time; it surfaced hours later as
# "PytorchStreamReader failed reading zip archive", and by then the worker --
# and /tmp with it -- was long gone, so two hours of training were lost.
#
# Checking free space BEFORE training would not have caught it: there was
# plenty at launch, and about a dozen people share the drive.  What catches it
# is verifying afterwards and staying alive while retrying:
#
#   * somebody else's job finishing unblocks it with no action at all;
#   * the retry window keeps this worker -- and therefore /tmp -- alive long
#     enough to clear space by hand.
#
# The price is idle GPU time on failure, capped by COPY_TRIES.  Two hours of
# that is cheap against two hours of training thrown away.
COPY_TRIES=${COPY_TRIES:-120}          # x 60s = 2 hours

copy_back() {                          # src dst
  local src=$1 dst=$2 want got i
  want=$(stat -c%s "$src")
  for i in $(seq 1 "$COPY_TRIES"); do
    mkdir -p "$(dirname "$dst")" 2>/dev/null || true
    cp -f "$src" "$dst" 2>/dev/null || true
    got=$(stat -c%s "$dst" 2>/dev/null || echo 0)
    if [ "$got" = "$want" ]; then
      echo "  copied $want bytes -> $dst"
      return 0
    fi
    {
      echo "  INCOMPLETE COPY ($i/$COPY_TRIES): $got of $want bytes -> $dst"
      df -h "$(dirname "$dst")" 2>/dev/null | tail -1
      echo "  source is still at $src on THIS worker; free some space and it"
      echo "  retries in 60s.  Losing it needs the task to be reclaimed first."
    } >&2
    sleep 60
  done
  echo "  GAVE UP on $dst after $COPY_TRIES attempts ($((COPY_TRIES)) minutes)." >&2
  return 1
}

# One file at a time rather than `cp -r`, so every checkpoint is verified on
# its own; a partial directory copy is exactly what went unnoticed before.
copy_fail=0
for f in "$OUT"/*; do
  [ -f "$f" ] || continue
  copy_back "$f" "$REPO/out_hihr_$MODE/$(basename "$f")" || copy_fail=1
done

if [ "$copy_fail" = 0 ]; then
  echo "done. checkpoint -> $REPO/out_hihr_$MODE ; log -> $LOG"
else
  {
    echo
    echo "TRAINING FINISHED BUT THE CHECKPOINT DID NOT SURVIVE THE COPY."
    echo "Every number is still in $LOG -- only the weights are at risk, and"
    echo "they are only needed for feature diagnostics, not for the results."
  } >&2
  exit 1
fi
