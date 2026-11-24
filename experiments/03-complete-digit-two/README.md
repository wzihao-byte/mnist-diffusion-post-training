# 手写数字 2 笔画补全实验

这个版本针对手写「2」的左上起笔、左下转折与右下尾笔。模型在原生 32×32 画布上生成图像，交付程序检查笔画结构并按需重抽。验收后的图片保留模型生成的原始像素。

## 实测结果

用一组训练种子和三组留出采样种子，在红绿蓝、左右位置的六个组合上评估。每个检查点各生成 1,152 张，阶段之间复用噪声。表格里每张图按一次原始采样计数，尚未重抽。

| 检查点 | 左上起笔可见 | 左下转折可见 | 右下尾笔可见 | 通过完整验收和独立 OCR |
| --- | ---: | ---: | ---: | ---: |
| 旧 32×32 | 72.6% | 99.6% | 87.8% | 40.8% |
| 完整笔画 SFT | 96.4% | 99.0% | 95.3% | 73.1% |
| 专项形状 RL | 98.5% | 99.8% | 97.8% | 82.2% |
| 专项 OPD | 97.3% | 99.5% | 96.6% | 76.0% |

专项 RL 在用于选权重的采样组上通过率最高，成为默认检查点。留出评估中有 947 张原始图通过交付验收，独立评估器也把这 947 张都识别为「2」。另运行 24 次默认推理请求，每次交付 4 张。共交付 96 张，全部通过笔画复验且被独立评估器识别为「2」。这些指标衡量所定义的字形结构及数字识别，样图用于查看具体书写风格。

完整记录见 `reports/release/audit.json`、`reports/release/samples.csv` 和 `reports/delivery.json`。对照图中的右侧绿色「2」第一张仍有闭环，在默认交付路径中自动换成同一序列的第二张。

## 用已有权重生成

使用 Python 3.12。演示和推理只读取附带的权重。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_two.py --prompt 'green 2 on the right' --count 6 --out demo/green.png
python verify_two.py --prompt 'red 2 on the left' --count 6 --out demo/red.png
```

支持红、绿、蓝三色和左右两个位置。`--raw` 会展示当前检查点的首批原始输出。`--max-draws` 控制重抽上限，默认 72。验收图不足时，命令明确返回合格数量与尝试次数。可用 `--device cpu` 在 CPU 上运行，也可让程序自动选择 MPS 或 CUDA。

## 训练和交付检查

1. 从原有 32×32 模型起步，在 MNIST 的 4,807 个训练「2」中按三处笔画的强度和连通性筛出 1,909 个字形。用它们继续做 SFT，并加相邻像素差损失。
2. 从 SFT 检查点进行 240 步形状 RL。奖励保留可读性、位置与颜色，并提高各笔画段同时出现、相互连通的收益。整张图的增强奖励由原奖励分类器的数字置信度调节。
3. 用同一组字形适配较宽教师，沿学生自己的轨迹执行 160 步 OPD。对比 SFT、RL、OPD 后，用专项 RL 的权重交付。
4. 推理时以 32×32 可见笔画像素检查左上起笔、左下转折、右下收尾、底笔覆盖和连通性。右端亮像素需到达字形宽度的 82%，并且距上半部最右端至多差一像素。另检查起笔和底笔附近的闭环、四边净空，以及数字、颜色和位置。

独立 OCR 评估器的权重来自上一轮对照，它的训练样本与专项生成训练分开。它用于报告中的独立复核。程序重抽时使用笔画规则以及另一套奖励分类器。

## 训练复现

```bash
python complete_two.py --data-dir data --run-dir reproduced
python audit_two.py --run-dir reproduced --split selection --repeats 48 --out reproduced_selection
python audit_two.py --run-dir reproduced --split final --repeats 64 --gallery --out reproduced_final
python check_delivery.py --run-dir reproduced --out reproduced_final/delivery.json
```

首次训练会下载公开 MNIST 并检查文件校验和。`reference/` 附上旧 32×32 起点、教师、训练奖励分类器与独立评估器。`run/` 有已训练权重和运行日志。对照图中的前四列为同一提示与噪声下的第一张原始图，最后一列为默认模型在该序列中第一张合格图。

[MNIST 数据](https://github.com/cvdfoundation/mnist) · [DDPO](https://rl-diffusion.github.io/) · [DiffusionOPD](https://arxiv.org/abs/2605.15055)
