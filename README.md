# Baseline / Trajectory 配对实验

本项目只保留 WHU-MARS 上的 CLIP ViT-B/16 Baseline、原 dense cross-layer token Trajectory，以及预留的四格 VTC。当前实验不启用 VTC。

2026-10-01 本轮矩阵为 **2 方法 × 2 normalization × 2 解释器环境 = 8 次独立训练**，每组 60 epoch，再从同一 `transformer_60.pth` 分别测试 pre-BN 和 post-BN，共 16 份最终测试结果。

- [实验方案与结果选择规则](0.八组配对实验方案_1001.md)
- [服务器预检与启动手册](1.服务器实验启动手册_1001.md)
- [代码整理与本地验证记录](2.代码整理与验证记录_1001.md)

| ID | 环境 | normalization | 模型 |
|---|---|---|---|
| W_S_B | whu_mars | symmetric 0.5/0.5 | Baseline |
| W_S_T | whu_mars | symmetric 0.5/0.5 | Baseline+Trajectory |
| W_C_B | whu_mars | CLIP native | Baseline |
| W_C_T | whu_mars | CLIP native | Baseline+Trajectory |
| L_S_B | llmpar | symmetric 0.5/0.5 | Baseline |
| L_S_T | llmpar | symmetric 0.5/0.5 | Baseline+Trajectory |
| L_C_B | llmpar | CLIP native | Baseline |
| L_C_T | llmpar | CLIP native | Baseline+Trajectory |

每组完整配置在 `configs/<ID>.yml`，独立前台启动脚本在 `server/run_<ID>.sh`。正式训练入口是 `finetune.py`，独立测试入口是 `test.py`。预检单独执行，训练结束自动串联 pre-BN 和 post-BN 测试。

配置检查：

```bash
/mnt/cache/wanghanzhi/envs/whu_mars/bin/python3 /mnt/cache/wanghanzhi/XK/Trajectory/tools/verify_matrix.py
```

服务器代码目录固定为 `/mnt/cache/wanghanzhi/XK/Trajectory`。本轮整理不代表代码已上传服务器、预检已通过或正式训练已完成。训练产物保留在忽略的 `logs/` 下，旧源码可通过 Git 历史恢复。
