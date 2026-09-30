# HiHR baseline, reproduced — then carried onto WHU-MARS

Two legs:

1. **`cargo_base`** — reproduce the baseline of *HiHR: Hierarchical Hyperbolic
   Representation for Aerial-Ground Person Re-Identification* (Yang & Zhang,
   ECCV 2026, arXiv:2607.09186) on **CARGO**. Target: table 4's `Baseline` row,
   **63.61 mAP / 68.23 Rank-1**.
2. **`whu_from_cargo`** — train on **WHU-MARS** starting from leg 1's weights.
   That result is the project's baseline going forward.

The authors' repository (`github.com/YangQiWei3/HiHR`) is a 404, so there is no
released code or checkpoint — leg 1 trains it from the paper.

Only the baseline is reproduced. MFE / TMF / HHL are not implemented: table 4
puts all three together at +3.9 mAP on CARGO, and HHL alone needs a Lorentz
manifold with entailment cones.

Built on the WHU-MARS UAD codebase (itself a TransReID derivative).

---

## Requirements

| | |
|---|---|
| Datasets | [CARGO](https://github.com/LinlyAC/VDT-AGPReID) · [WHU-MARS](https://github.com/msm8976/WHU-MARS#whu-mars-dataset) |
| Weights | [CLIP ViT-B/16](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) — the hex in that URL is the file's sha256 |

CARGO's released split is **102,475 images**, not the 108,563 both papers quote.
VDT section 7.1 keeps only two cameras per query identity, so 6,088 images are
dropped by the protocol. Expected: train 51,451 / 2,500 ids; query 312 / 149
ids; gallery 50,712.

---

## Running

```bash
bash verify_upload.sh              # dev machine, ~1 min, no GPU
bash run_hihr.sh cargo_base        # GPU worker, ~1.5 h
bash run_hihr.sh whu_from_cargo    # GPU worker, ~1.5 h
```

`verify_upload.sh` checks the *contents* of the files that had to change (not
just that they arrived), runs the five unit test suites, verifies the CLIP
sha256, checks both dataset layouts, and loads the config through yacs to print
what training will actually see. Everything in it is CPU-only on purpose: one
minute on the dev machine rules out the mistakes that otherwise cost an hour and
a half of GPU time.

`check_cargo.py` is a one-off for a fresh CARGO download: it decodes all 102,475
images with `LOAD_TRUNCATED_IMAGES` turned **off**, because `datasets/bases.py`
turns it on and a truncated JPEG would otherwise train silently as grey mush.

`reeval.sh` re-scores an existing checkpoint under a different test-time setting
without retraining.

### What the logs must say

```
Loaded 153 CLIP visual tensors; 2 model tensors left at init: ['fc.bias', 'fc.weight']
Base Lr: 3.50e-04 (pretrained 5.00e-06)
```

and for leg 2, `loaded 160 / 161 tensors | 1 classifier rows skipped`. Anything
else there means the weights did not fully load, which is the entire point of
that leg.

---

## What changed from the UAD codebase

**`vit_base_clip`** — CLIP's image tower differs from a plain ViT in four ways,
none of which raise if you get them wrong; they just cost accuracy: QuickGELU
rather than `nn.GELU`, an extra `ln_pre` before block 0, no bias on the patch
convolution, and LayerNorm `eps=1e-5`. `tests/test_clip_backbone.py` pins this
by running a transcription of CLIP's own `VisionTransformer.forward` on the same
weights and demanding the outputs agree.

**`TEST.METRIC`** — both evaluation functions used to hardcode the SYSU rule
(drop the query's entire camera). CARGO and the AG-ReID benchmarks score with
Market-1501 (drop only the query's own identity from that camera), which reads
several points lower. WHU-MARS keeps `sysu`; CARGO uses `market`.

**`SOLVER.PRETRAINED_LR` / `WARMUP_ITERS`** — HiHR runs the pretrained tower at
5e-6 against 3.5e-4 for freshly initialised modules, and warms up over 100
*iterations* (0.05 of an epoch here, which epoch-granularity stepping cannot
express). `CosineLRScheduler` took a scalar floor, which would have given the
two tiers differently shaped curves; it now takes one per group.

**`MODEL.PRETRAIN_CHOICE`** — used to have a single `'imagenet'` branch and fell
through in silence for anything else, training from random init. Now `'self'`
(a checkpoint from this codebase, keys prefixed `base.`, classifier skipped
because identity counts differ), `'no'`, or an error. The `'self'` load runs at
the very end of `__init__`, since `bottleneck` and `classifier` do not exist
before then.

**`datasets/cargo.py`** — the four protocols (ALL / AA / GG / AG). Note AA and
GG filter the *training* set too, so re-scoring an ALL-trained checkpoint under
them is not the paper's number; AG is the only sub-protocol whose training set
is the full one. Filename parsing is copied from the official loader rather than
inferred — an off-by-one in the underscore index relabels the whole dataset and
still trains happily.

CARGO is single-modality and this pipeline was written for three, but no fork
was needed: every stage is parameterised by a modality list, so CARGO reports
itself as one pseudo-modality and `PKMSampler` degenerates to plain PK.

---

## Tests

```bash
for t in test_cargo test_cargo_pipeline test_transfer_weights \
         test_hihr_recipe test_clip_backbone; do
  python tests/$t.py | tail -1
done
```

44 tests, CPU-only, no dataset and no pretrained weights required.

---

## Results

| | mAP | Rank-1 |
|---|---|---|
| CARGO, paper's table 4 `Baseline` | 63.61 | 68.23 |
| CARGO, `cargo_base` | **62.67** | **68.27** |

Five hyperparameters the paper does not state are marked
`paper does not state` in `configs/hihr_cargo_base.yml`: weight decay, the
triplet margin, the cosine floor, label smoothing, and the padding before the
random crop.
