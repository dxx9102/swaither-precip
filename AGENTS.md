# AGENTS.md

## 项目背景

本项目由一个瑞士降雨超分项目扩展而来，在原有降雨超分任务基础上增加了风场预测与风场超分相关功能。

## Python 环境

本项目使用 Anaconda 管理 Python 环境。Anaconda 本身通过 `module` 管理，**不能假设终端中已经可以直接使用 `conda`**。

在本项目目录下执行任何 Python 相关操作前，必须先完成以下三步环境加载与激活：

```bash
module load anaconda3/2023.09
eval "$(conda shell.bash hook)"
conda activate swaither-precip
```

## 环境使用规则

- 所有 Python、pip、pytest 或其他依赖该环境的命令，都应在 `swaither-precip` 环境中执行。
- 如果发现 `conda` 命令不可用，应先执行 `module load anaconda3/2023.09`。
- 如果 `conda activate` 报错，应先执行 `eval "$(conda shell.bash hook)"`，再重新激活环境。
- 不要假设当前 shell 已经加载了 Anaconda，或已经激活了 `swaither-precip` 环境。
- 在安装依赖、运行训练、推理或测试前，先确认当前 Python 解释器来自 `swaither-precip` 环境。

## 环境验证

激活环境后，可使用以下命令确认环境是否正确：

```bash
which python
python --version
conda env list
```

当前应处于 `swaither-precip` 环境中。

## 代码测试与验证

- 修改代码后应进行必要的测试或验证，确认改动不会破坏现有功能。
- 对于不依赖 GPU 的小规模测试，可在激活 `swaither-precip` 环境后直接在登录节点执行。
- **不要在 CPU 端进行大规模或其他耗时的测试，包括大规模数据加载和计算等；此类测试应尽量放到 GPU 端进行。**
- 对于需要使用 GPU 的测试或验证，**不得直接在登录节点运行**，必须通过 `yhbatch` 提交作业。

## GPU 测试作业提交规则

如果需要使用 GPU 相关资源，必须遵循以下流程：

1. **先编写 `.sh` 脚本**，将需要在 GPU 上执行的命令写入脚本中。
2. **使用 `yhbatch` 提交该脚本**，不得直接在终端中运行 GPU 命令。
3. 提交时**选择分区 `a800x_595`**，并**申请 1 个 GPU**。
4. 使用 `-o` 指定作业输出文件的路径。

标准提交方式如下：

```bash
yhbatch -p a800x_595 -G 1 -o ./xxx.out xx.sh
```

参数说明：

- `-p a800x_595`：指定分区为 `a800x_595`。
- `-G 1`：申请 1 个 GPU。
- `-o ./xxx.out`：指定记录测试脚本输出的文件路径。
- `xx.sh`：待提交的作业脚本。

## `.sh` 脚本中的工作目录要求

通过 `yhbatch` 提交的 `.sh` 脚本，其运行时的工作目录不会自动继承提交时的当前目录。因此，在脚本中必须**显式指定工作目录**，例如使用 `cd` 切换到项目目录后再执行命令。

具体写法请参照工程目录下的 `./scripts/yhbatch_xxx.sh` 系列脚本。编写新的 GPU 测试脚本时，应优先复用或参考这些脚本的结构与目录切换方式。

## GPU 测试流程小结

```text
编写 .sh 脚本（显式 cd 到工作目录）
        ↓
yhbatch -p a800x_595 -G 1 -o ./xxx.out xx.sh
        ↓
查看 ./xxx.out 中的作业输出与测试结果
```