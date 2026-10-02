# Trajectory：Baseline / Trajectory 严格对照

## 第三轮：Token分工与更新来源（2026-10-02）

新增P1（仅CLS加速度）、P2（仅patch加速度）、P3（Attention残差速度）、P4（MLP残差速度）。四组沿用原始dense gain，固定post-BN主结果、pre-BN补充；旧baseline与全部C/T/R组保留。各组配置位于`configs/trajectory_P*.yml`，独立单卡入口位于`server/structure/`。

见 [第三轮方案与验证说明](doc/9.Trajectory第三轮Token分工与更新来源方案_1002.md) 和 [第三轮单卡启动手册](doc/10.Trajectory第三轮单卡启动手册_1002.md)。本地49项测试、完整CLIP构造检查、缩小模型AMP与八个脚本参数检查已通过；尚未进行本轮正式训练。

## 第二轮：弱加速度与门控调整（2026-10-02）

新增R1固定beta=0.5、R2自适应beta初值0.5、R3原T4门控无权重衰减、R4纯速度门控。旧baseline、T0、C0—T5保留。新组固定post-BN主结果、pre-BN补充，独立入口位于`server/refinements/`。

见 [第二轮四组方案与隔离说明](doc/6.Trajectory第二轮四组方案与隔离说明_1002.md) 和 [第二轮单卡启动手册](doc/7.Trajectory第二轮单卡启动手册_1002.md)。R1—R4的60epoch训练及双读出测试已完成，详见 [第二轮结果分析](doc/8.Trajectory第二轮结果分析_1002.md)：四组尚未超过第一轮C0/T1/T2/T4的总体结果；R2最终beta接近1，R3/R4门控权重不再近零但未提高性能。历史T0继续沿用，当前只分析seed1234。

## 第一轮：Trajectory 模块扩展（2026-10-02）

固定 llmpar + CLIP norm + cuDNN benchmark=True / deterministic=True，保留原 dense/direct 为T0。新增T1自适应加速度、T2双分支、T3历史平滑、T4样本门控、T5通道混合，以及C0速度对照。各版独立配置为`configs/trajectory_*.yml`；SenseCore/AutoDL独立前台入口位于`server/extensions/`，训练后自动补测同一epoch60权重的pre-BN/post-BN。

见 [五版实验方案](doc/2.Trajectory五版扩展实验方案_1002.md) 和 [单卡启动手册](doc/3.Trajectory扩展单卡启动手册_1002.md)。C0及T1—T5的首轮seed1234训练与双读出测试已完成，详见 [首轮结果分析](doc/5.Trajectory扩展首轮结果分析_1002.md)。C0的pre-BN mAP/Rank-1为13.13%/33.74%。T0前6epoch日志与历史记录一致，本阶段按用户决定沿用历史T0、保留原参考实现，不补完整重训或多seed。

## 历史轮：环境与输入 normalization 对照

当前训练与测试链路恢复到本窗口已测试的 `0158be2`，不沿用后续重构；原 Trajectory、VTC 实现与日志格式不改。

本轮八组实验为：Baseline / Baseline+Trajectory × symmetric / CLIP native normalization × llmpar / whu_mars。统一 seed=1234、60 epoch、cuDNN benchmark=True、deterministic=True，VTC 全部关闭。每组训练后用同一个 `transformer_60.pth` 分别测试 pre-BN（主结果）、post-BN（补充结果）。

配置在 `configs/{L,W}_{S,C}_{B,T}.yml`，独立启动入口在 `server/run_{L,W}_{S,C}_{B,T}.sh`。无需强制预检；脚本调用原始 `train.py`、`test.py`。

详细设置、八组启动命令、仅测试命令与回退说明见 [实验方案与启动命令](doc/0.八组最小改动实验方案与启动命令_1001.md)。

保留 `configs/whu_trajectory_only.yml`、`configs/whu_trajectory_vtc.yml` 和 `run_trajectory.sh`，供历史复现及以后 VTC 实验使用；不混入本轮结果。
