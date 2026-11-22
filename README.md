# FTsaki-1118

FTsaki 的工业训练 Harness，交付版本使用早期 16 步 LoRA 初训模型 `model0-1118`。系统按比赛 SDK 1.0.2 的 `contest.harness.workspace.v2` 协议接收公开训练与验证数据，在 CPU 上续训工业模型，经比赛宿主网关由决策模型选择候选或父产物，再用选定检查点逐条预测。

| 场景 | 工业模型输出 |
| --- | --- |
| A · 气体识别 | 六个气体类别及分数 |
| B · 轴承寿命 | 剩余分钟数 |
| C · 发动机寿命 | 剩余循环数 |

## 构建与检查

在有 Docker 权限的 Linux amd64 环境中，从仓库根目录运行以下命令。首次构建需要拉取固定摘要的 Python 3.12 基础镜像。构建过程中使用仓库自带的 wheel 安装 NumPy 1.26.4 与构建依赖，并校验 18 个 CPU 基线文件。

```sh
sh contest/build.sh
docker run --rm --network none --read-only ftsaki-contest:model0-1118 describe
```

`describe` 应报告 `contest.harness.workspace.v2`、`host_gateway_rpc`，以及 `classification` 和 `regression` 两类任务。

## 复现本地验收

准备赛事分发的 A/B/C 公开 CPU 开发包，将三个解压目录放在同一父目录下。该目录作为参数传给验收脚本。

```sh
python3 contest/verify_public.py /path/to/data/sources
python3 contest/verify_public.py /path/to/data/sources --full
```

默认每个场景取 64 条公开训练记录和 16 条公开验证记录，`--full` 使用完整公开集。三个场景已通过完整公开集验收。测试在断网的只读容器里实际续训并提交候选，随后分别用父、子检查点预测。模型回复、测量及回合确认由测试程序模拟，真实模型回执和评测结果由赛事平台产生。

比赛平台使用持久 Sampler `trio://run_ogvplibsgt3s/sampler_weights/trial-final`，对应检查点 `ckpt_94d6jwg93fpj`。在平台上完成模型绑定及凭据配置，容器由宿主网关调用该模型。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [contest/](contest/) | 比赛训练与预测入口、Dockerfile、构建及验收脚本 |
| [baselines/](baselines/) | A/B/C CPU 基线算法和配置 |
| [src/ftsaki_harness/](src/ftsaki_harness/) | 漂移监测、模型池、回放和实验记录 |
| [config/](config/) · [scripts/](scripts/) | 开发配置、微调验收、回放和结果导出 |
| [ModelTraining/](ModelTraining/) | 初训、输入投影、评估代码和历史实验配置 |
| [data/](data/) | 数据准备与校验脚本 |

历史模型训练依赖见 [requirements-training.txt](ModelTraining/requirements-training.txt)。复核历史训练需要原训练工程的配套数据和工作区。镜像构建与本地验收使用独立的运行环境。

NumPy 使用 BSD-3-Clause，setuptools 和 wheel 使用 MIT，PyTRiO 使用 MIT，SwanLab 和 Transformers 使用 Apache-2.0。赛事 SDK 和公开基线来自比赛分发包，使用范围依赛事授权。Python、Debian 基础镜像及二进制依赖的详细许可证随对应组件分发。
