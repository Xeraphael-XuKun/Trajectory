# Dense Cross-layer Token Trajectory + VTC

This is an independent, single-variable replacement of dense PLD in the
verified PLD+VTC experiment.

## Changed

- `MODEL.PE_LAYERWISE` is `none`: no `pos_delta` parameter is allocated.
- From layer 1 onward, the module observes the token displacement produced by
  preceding Transformer blocks.  It combines velocity and acceleration, then
  applies a zero-initialised dense token/channel gain.
- The module has 11 x 129 x 768 = 1,089,792 parameters and owns no static token
  feature offset.
- VTC disables trajectory correction in its `before` pass and enables it in
  its `after` pass.
- Trajectory gains use the same fresh-module learning rate and multiplier that
  PLD used.

## Unchanged

Four-cell VTC, learnable prompts, `TEXT_LOSS_WEIGHT=5.0`, CLIP-ViT-B/16,
CE+soft-margin Triplet, data sampling, augmentation, optimizer, learning-rate
schedule, epochs and evaluation protocol are unchanged.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/path/to/python3 \
CLIP_WEIGHT=/path/to/ViT-B-16.pt \
DATA_ROOT=/path/to/datasets \
OUTPUT_DIR=./logs/whu_trajectory_vtc \
bash run_trajectory.sh train
```

At initialisation all trajectory gains are exactly zero, so the image tower
equals the raw CLIP baseline.  Accuracy relative to PLD must be established by
the same 60-epoch protocol; it is not claimed without a completed run.
