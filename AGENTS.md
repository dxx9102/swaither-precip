# AGENTS.md

## 项目背景

本项目由一个瑞士降雨超分项目扩展而来，在原有降雨超分任务基础上增加了风场预测与风场超分相关功能。

## Python 环境

本项目使用 Anaconda 管理 Python 环境。Anaconda 本身通过 `module` 管理，**不能假设终端中已经可以直接使用 `conda`**。

在运行任何 Python、pip、pytest 或其他依赖该环境的命令之前，必须先在新终端或新 Codex 会话中执行以下命令：

```bash
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip