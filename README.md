# H-P3 + original M2-5 on A800

本分支是独立的实验源码快照，对应本地实现提交 `50fa9f0f1c70dd1d4a6cc01434c0219e017ad99a`。完整来源见 `SOURCE_PROVENANCE.json`。

保留原 H-P3，加入原 M2-5 条件教师蒸馏；学生从原 CLIP 初始化，使用原 A800 教师及身份文本产物。固定逻辑 batch64、seed1234、60epoch，epoch60 ALL post-BN 为主读出，同权重 pre-BN 为补充。

- 配置：`configs/hp3_m25_a800.yml`
- 独立预检：`server/hp3_m25_a800/preflight.sh`
- 前台训练及末次双读出评测：`server/hp3_m25_a800/run_train.sh`
- 单独评测：`server/hp3_m25_a800/eval_epoch60.sh`

脚本使用 `/mnt/cache/wanghanzhi/XK/Trajectory_hp3_m25_a800_1018` 代码目录及原 `llmpar` 解释器。权重、数据、教师和文本产物需要在配置所列路径已有，不随源码提供。

状态：本地学生数值检查已完成；原教师的 CPU 严格加载未通过。保持生产严格加载检查，A800 预检、正式训练及检索评测尚未执行。
