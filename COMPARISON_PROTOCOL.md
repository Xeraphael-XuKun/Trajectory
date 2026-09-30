# PLD / ATR / Token-Trajectory fair comparison

Use the same official `ViT-B-16.pt`, WHU-MARS split, GPU count and seed for all
three arms.  Do not reuse a PLD-trained checkpoint: every arm starts from the
same raw CLIP checkpoint and trains for 60 epochs.

| Arm | Positional actuator | New parameters | Config |
|---|---|---:|---|
| Verified reference | dense independent PLD | 1,188,864 | `configs/hihr_whu_text_lam50_clip.yml` |
| ATR | dense attention edge residual | 1,198,152 | `configs/whu_atr_vtc.yml` |
| Trajectory | dense velocity/acceleration gain | 1,089,792 | `configs/whu_trajectory_vtc.yml` |

The comparison configs differ from the verified reference only in the active
actuator and `OUTPUT_DIR`.  VTC, its weight 5.0, all identity losses, data,
augmentation, optimizer, schedule and evaluation remain unchanged.

Report at least:

| Arm | mAP | Rank-1 | Rank-5 | A->G mAP | G->A mAP | G->G mAP |
|---|---:|---:|---:|---:|---:|---:|
| PLD+VTC | 14.11 | 36.14 | 53.38 | 13.85 | 16.03 | 15.18 |
| ATR+VTC |  |  |  |  |  |  |
| Trajectory+VTC |  |  |  |  |  |  |

Also inspect every RGB/NIR/Thermal query-gallery cell.  A global mAP gain that
comes only from one same-spectrum diagonal cell is not evidence that the new
actuator improves multispectral aerial-ground alignment.
