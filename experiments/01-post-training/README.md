# 彩色数字扩散模型后训练实验

输入 `blue 7 on the left`，模型从随机噪声开始生成一张 16×16 RGB 图像。数字、颜色和位置是可独立验证的指令。本项目在 Apple M4 的 MPS 上训练了一个条件扩散模型，运行监督微调、基于奖励的强化学习和学生轨迹上的蒸馏，并对比各阶段的检查点。

## 使用已有权重

解压后进入项目目录，使用 Python 3.12。已附带训练好的模型和评估器，推理时可离线运行。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python evaluate.py --prompt 'blue 7 on the left' --stage rl_opd --out-dir demo
```

示例写入 `demo/sample.png`。提示词格式为 `<red|green|blue> <0–9> on the <left|right>`。可用 `--stage sft`、`--stage rl`、`--stage opd`、`--stage rl_opd` 或 `--stage teacher` 对比检查点。默认选择 CUDA、MPS 或 CPU，也可用 `--device cpu` 显式指定。

## 从头训练与重新评估

```bash
python train.py --run-dir reproduced --data-dir data
python evaluate.py --weights reproduced --out-dir reproduced_results
```

训练首次下载 CVDF 上的 MNIST 压缩文件并验证校验和。已完成的阶段会复用该目录的检查点。中断的阶段从该阶段开头重跑。也可用 `python train.py --smoke --run-dir smoke` 快速验证各阶段能跑通。跨 PyTorch 设备的数值和结果可能略有差异。

主实验的默认种子为 143。复跑补充两个种子并计算平均值的方法如下。

```bash
for seed in 144 145; do
  python train.py --seed "$seed" --run-dir "reproduced_$seed" --data-dir data
  python evaluate.py --weights "reproduced_$seed" --out-dir "results_$seed"
done
python summarize.py reproduced_results/metrics.json results_144/metrics.json \
  results_145/metrics.json --out reproduced_results/multi_seed.json
```

## 数据与训练

从 MNIST 训练集用固定随机种子划分 48,000 张生成训练图、6,000 张奖励分类器训练图和 6,000 张独立评估分类器训练图。原始字形缩为 12×12，通过渲染赋予三种颜色和两个位置。原始 10,000 张 MNIST 测试图只用于检查两个分类器的识别表现。

```text
无条件 Base → SFT ─┬→ RL → OPD
                    └→ OPD
更宽的条件 Teacher ───→ 向两条 OPD 路径提供去噪目标
```

- **SFT** 从无条件预训练权重开始，加入数字、颜色和位置三个嵌入。对加噪后的真实图预测干净图 `x₀`，用像素均方误差做监督。训练用 16 步的固定方差 DDPM。
- **RL** 从 SFT 取样。冻结的数字分类器给指定数字打分，并用颜色占比、亮度重心和墨迹量计算终局奖励。同一提示词抽四张，以其余三张的平均奖励构造优势。用 DDPO 启发的 REINFORCE 更新随机去噪转移的对数概率，加入指向 SFT 的高斯 KL 正则。每轮重新采样，最后一步是确定性的，因此只在有方差的转移上计算策略梯度。
- **OPD** 每轮先从当前学生取样，再在实际访问到的中间状态上查询冻结的教师。以教师预测的 `x₀` 作为监督，优化学生与教师之间的均方误差。固定去噪步 `t` 时，未裁剪的 `x₀` 匹配与转移均值匹配成比例。这是单教师、固定方差 DDPM 下的简化实现。

## 评估协议

每个训练种子均评估全部 60 种提示词。每个提示词用两个推理种子，各采样八张，共 960 张每阶段。阶段之间使用相同提示和噪声，比较图固定展示主实验第一个推理种子的第一张样本。联合通过要求数字由独立分类器识别正确、目标色至少占有效颜色总量的 55%、亮度重心位于指定侧且离中线至少 0.5 像素，并有足够前景。单次评估差值区间使用配对样本 bootstrap，反映固定检查点的生成采样波动。

`results/metrics.json` 记录主实验汇总，`results/examples.csv` 提供逐样本结果，`results/comparison.png` 展示生成对照，`results/reward_disagreements.png` 展示高奖励的数字判分分歧。`checkpoints/training.jsonl` 保存训练过程，`checkpoints/config.json` 保存运行配置。补充两组种子的配置、日志和评估汇总位于 `results/replicas/`。

## 实测结果

在 Apple M4 上用三个训练种子运行，每阶段共生成和评估 2,880 张图。三组实验均使用同一配置，分别重新划分数据、训练生成器和两套分类器。对照图与附带权重来自种子 143。学生网络有 164,923 个参数，教师网络有 641,723 个参数。

| 检查点 | 种子 143 | 种子 144 | 种子 145 | 平均联合通过率 | 较 SFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无条件 Base | 1.25% | 1.46% | 0.63% | 1.11% | — |
| SFT | 83.12% | 85.42% | 83.12% | 83.89% | — |
| SFT → RL | 91.35% | 92.29% | 89.38% | 91.01% | +7.12 个百分点 |
| SFT → OPD | 91.98% | 93.23% | 90.00% | 91.74% | +7.85 个百分点 |
| SFT → RL → OPD | 90.62% | 92.40% | 91.35% | 91.46% | +7.57 个百分点 |
| 教师 | 93.65% | 94.79% | 94.58% | 94.34% | +10.45 个百分点 |

三个追加训练分支在三组种子上均高于 SFT。完整链路和直接 OPD 的均值相近，两条路径的相对次序随种子变化。各阶段的更新次数不同，表格用于比较实际训练出的检查点。

汇总原始数值见 `results/multi_seed.json`。

## 判分分歧

主实验完整链路有 90 张图未联合通过，其中 74 张通过了颜色和位置检查，但独立分类器给出的数字与提示不符。74 张中有 20 张的训练奖励至少为 0.90。奖励最高的一张要求右侧绿色 5，奖励为 0.993，独立分类器读成 8。图中也能看到接近目标数字的样本，说明自动分数会同时受生成质量和分类器误判影响。主实验两个分类器在原始 MNIST 测试集经同样渲染后的准确率分别为 94.79% 和 91.97%。

项目的受控任务是数字、颜色和左右位置，生成分辨率为 16×16。联合通过率衡量这三个条件，图像观感和笔迹多样性未做独立量化。首轮全连接去噪器主要输出色噪，后续同时改用了卷积 U-Net 和预测干净图像的目标函数。

## 代码入口

| 文件 | 作用 |
| --- | --- |
| `mnist_data.py` | 下载、核验与渲染 MNIST，定义奖励与评估用的分类器结构 |
| `diffusion.py` | 条件小型 U-Net、前向加噪、反向转移和学生轨迹采样 |
| `train.py` | 训练基线、SFT、教师、RL、OPD 与三阶段串联检查点 |
| `evaluate.py` | 单条提示推理、配对评估和对照图 |
| `summarize.py` | 读取多个评估 JSON，汇总各训练种子的联合通过率 |

## 公开资料

- [MNIST 数据与 CVDF 镜像](https://github.com/cvdfoundation/mnist)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Training Diffusion Models with Reinforcement Learning](https://rl-diffusion.github.io/)
- [DiffusionOPD](https://arxiv.org/abs/2605.15055)
