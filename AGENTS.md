# 项目级协作约定

## 1. 环境、数据、权重路径

本地路径：

    WHU-MARS-1000 数据集：E:\CFAN\datasets\WHU-MARS
    CLIP ViT 权重：E:\CFAN\weights\ViT-B-16.pt
    Pre-trained model 权重：E:\CFAN\weights\jx_vit_base_p16_224-80ecf9dd.pth

sensecore 云开发机路径：

    项目代码： /mnt/cache/wanghanzhi/XK/Trajectory
    已有 whu_mars 解释器： /mnt/cache/wanghanzhi/envs/whu_mars/bin/python3
    已有 llmpar 解释器： /mnt/cache/wanghanzhi/envs/llmpar/bin/python3
    WHU-MARS-1000 数据集：/mnt/cache/wanghanzhi/Datasets/WHU-MARS
    WHU-MARS-2337 数据集：/mnt/cache/wanghanzhi/Datasets/WHU-MARS-2337
    CLIP ViT 权重：/mnt/cache/wanghanzhi/Datasets/ViT-B-16.pt
    Pre-trained model 权重：/mnt/cache/wanghanzhi/Datasets/jx_vit_base_p16_224-80ecf9dd.pth

已检查的 `llmpar` 环境为 Python 3.8、torch 2.4.1+cu121、torchvision 0.13.1、timm 1.0.19。torch 与 torchvision 不配套，并出现 `libtorch_cuda_cu.so` 加载警告；同时官方 requirements 中的新版依赖不再支持 Python 3.8。

## 2. GPU 与服务器启动规则

后续可能使用到sensecore 高性能算力池ACP中的A800-80GB GPU或者autoDL中租借的GPU。视具体GPU空余情况可能会存在变动。主要使用sensecore 高性能算力池ACP中的A800-80GB GPU，当无可用A800 GPU时，暂时转换至autoDL上租借的GPU.

sensecore 高性能算力池ACP:

- 每次仅可使用单卡A800 GPU进行训练。
- 每个实验必须有独立启动命令，并使用单卡环境：CUDA_VISIBLE_DEVICES=0、WORLD_SIZE=1。
- 正式服务器启动脚本应前台直接执行完整 finetune.py；不要用 nohup ... & 作为正式启动方式。预检和完整训练启动分开，不能用预检通过代替训练已启动。
- 启动命令须使用服务器上的真实绝对路径、明确的 Python 解释器、数据集、权重、输出目录和全部关键超参数。避免未定义 shell 变量、HTML 转义路径和被错误转义的选项名。
- 服务器实验手册和训练脚本的叙述使用中文；命令、路径、文件名、参数名和数值保持可直接复制，不翻译或改写。

autoDL:

后续租借时补充说明。

## 3. 避免代码防御性编程

注意不要过度编程，不要把简单问题复杂化，不要防御性编程。仅做保证数值结果可信所必需的sanity checks，不要把主要精力用于单元测试覆盖率、hash、格式化、CI、异常输入防御等，不要进行无意义的SHA256校验，仅仅在必要时才校验。

## 4. Git、文件和协作纪律

- 每次项目大修改后建立一次 Git 提交；小幅改动/某些修改补充可不要求单独提交。除非用户另行明确要求，不推送 GitHub、不上传服务器代码，也不发布实验结果。
- 不把训练产物、checkpoint、日志和本地数据加入git仓库。

## 5.实验对比方案与服务器启动手册

每当要求给出一份实验对比方案文档与服务器实验启动手册时，注意使用markdown文档形式，文档中写明实验设置、参数配置与启动命令，均使用中文，服务器实验启动手册与服务器训练脚本中的路径可直接写入服务器对应路径。

## 6.文档命名规范

文档命名建议采用“序号.内容概括_日期.md”格式，序号按照顺序递增。例如：0.复现实验顺序_0919.md。

## 7.方案讨论注意事项

- 不要默认认同我的观点，也不要为了让我满意而附和我，最重要的是帮助我发现自己忽略/错误的问题。
- 回答前，先检查我的问题里是否存在错误前提、逻辑跳跃、信息缺失或未经证实的判断。如果有，请直接指出，不要沿着错误前提继续推导。
- 请明确区分：已确认的事实、合理推测、个人观点和暂时无法验证的信息。涉及事实、数字、人物、日期、引用和案例时，尽可能核对来源；无法确认就直接说明，不要编造。
- 当你不同意我的判断时，请清楚说明原因，并给出反例、风险或更合理的解释。不要只给结论，要说明推理过程。

<!-- EXPERIMENT_DOCUMENT_ARCHIVE_RULE -->
## 8. 实验文档生成后的增量归档

- 本地生成或更新实验方案、结果说明/分析、实现与实验手册、综合实验汇总后，完成内容核对，必须调用 `E:\CVPR2027\ref\Trajectory\tools\archive_experiment_documents.py` 复制归档，然后再交付。用户已授权这一常规归档步骤，不需每次重新询问。
- 默认目标为 `E:\CVPR2027\ref\实验方案与结果汇总`。明确选择“方案日期_任务”批次，同一任务后续结果沿用原目录，不按结果生成日期另建批次；模块一与模块二同名组分开。现有批次映射见 `E:\CVPR2027\ref\Trajectory\doc\40.实验文档增量自动归档使用说明_1015.md`。
- 调用格式：`python <上述脚本绝对路径> --batch <日期_任务> --kind <文档类型> --status <实际执行状态> <原始文档绝对路径...>`。使用可用本地Python3.8以上解释器，不安装新依赖。失败或未完成状态如实记录，不把模板/手册当正式结果。
- 原文保留，文件名保留；相同内容跳过，更新前由脚本保存旧副本至批次内 `_历史版本`。脚本维护根目录索引和来源CSV。只归档md/pdf/docx文档，不复制训练产物或代码。
- 归档权限不足时走工具权限机制；如仍不能写入，交付时明确说明未归档及原因，不将未执行写成成功。服务器生成文档先下载到本地后归档，不在服务器使用Windows路径。
- 新实验工作树开始文档工作前，同步这一归档约定并读取批次说明；其他软件或手动修改不会触发后台同步，此规则由生成/更新文档的助手执行。
<!-- END_EXPERIMENT_DOCUMENT_ARCHIVE_RULE -->
