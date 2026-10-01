# Trajectory：Baseline / Trajectory 严格对照

## 当前轮：Trajectory 模块扩展（2026-10-02）

固定 llmpar + CLIP norm + cuDNN benchmark=True / deterministic=True，保留原 dense/direct 为T0。新增T1自适应加速度、T2双分支、T3历史平滑、T4样本门控、T5通道混合，以及C0速度对照。各版独立配置为`configs/trajectory_*.yml`；SenseCore/AutoDL独立前台入口位于`server/extensions/`，训练后自动补测同一epoch60权重的pre-BN/post-BN。

见 [五版实验方案](doc/2.Trajectory五版扩展实验方案_1002.md) 和 [单卡启动手册](doc/3.Trajectory扩展单卡启动手册_1002.md)。本地数值检查已完成，性能待正式训练验证。

## 历史轮：环境与输入 normalization 对照

当前训练与测试链路恢复到本窗口已测试的 `0158be2`，不沿用后续重构；原 Trajectory、VTC 实现与日志格式不改。

本轮八组实验为：Baseline / Baseline+Trajectory × symmetric / CLIP native normalization × llmpar / whu_mars。统一 seed=1234、60 epoch、cuDNN benchmark=True、deterministic=True，VTC 全部关闭。每组训练后用同一个 `transformer_60.pth` 分别测试 pre-BN（主结果）、post-BN（补充结果）。

配置在 `configs/{L,W}_{S,C}_{B,T}.yml`，独立启动入口在 `server/run_{L,W}_{S,C}_{B,T}.sh`。无需强制预检；脚本调用原始 `train.py`、`test.py`。

详细设置、八组启动命令、仅测试命令与回退说明见 [实验方案与启动命令](doc/0.八组最小改动实验方案与启动命令_1001.md)。

保留 `configs/whu_trajectory_only.yml`、`configs/whu_trajectory_vtc.yml` 和 `run_trajectory.sh`，供历史复现及以后 VTC 实验使用；不混入本轮结果。
