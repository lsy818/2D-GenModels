# 2D-GenModels

一个用于研究二维复杂分布生成建模的可复现实验框架。项目围绕 Gaussian Mixture、Ring、Two Moons 和 Spiral 四类数据，比较 VAE 与 DDPM 在多峰、薄流形、非凸结构和螺旋拓扑上的生成能力，并提供诊断实验、结构增强实验和论文级可视化。

## 特性

- 支持四类二维基准分布的数据生成与隐藏测试集构造。
- 实现 VAE / beta-VAE 与 DDPM 两类主生成模型。
- 提供 MMD、Wasserstein、Coverage、Precision 等生成质量指标。
- 包含条件生成、异常点鲁棒性、VAE 潜空间诊断、beta sweep 和 Spiral 结构增强实验。
- 自动生成 PDF 图表，便于论文、报告和实验复现实用。

## 项目结构

```text
2D-GenModels/
├── code/                  # 可执行代码与模型实现
├── data/                  # 数据集与标签
├── figures/               # 自动生成的实验图
├── models/                # 训练得到的模型权重
├── output/                # 实验指标 JSON
├── report/                # LaTeX 报告与编译产物
├── requirements.txt       # Python 依赖
└── README.md              # 项目说明
```

更详细的脚本结构与命令说明见 [code/README.md](code/README.md)。

## 快速开始

推荐使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

生成数据：

```bash
python -m code.generate_data
```

训练主模型、运行扩展实验并生成全部图表：

```bash
python -m code.run
```

如果已经有模型和 `output/*.json`，只重新生成图表：

```bash
python -m code.run --figs-only
```

## 实验输出

主要输出目录如下：

```text
data/                训练集、测试集、隐藏测试集与标签
models/              VAE 与 DDPM 主模型权重
models/extensions/   诊断和结构增强实验模型
figures/analysis/    数据几何分析图
figures/generation/  生成样本与扩散过程图
figures/evaluation/  主实验、条件生成与鲁棒性评价图
figures/extensions/  VAE 诊断、调参和 Spiral 增强图
output/              实验指标 JSON
```

## 方法概览

主实验分别在四类分布上训练 VAE 和 DDPM，并用统一指标评估生成质量。诊断实验进一步分析 VAE 的 KL 压缩、潜空间利用和先验采样偏差；对于 Spiral 这类全局拓扑更强的数据，项目额外实现了结构坐标增强实验，用显式的螺旋臂、相位和局部残差变量改善生成质量。

## 复现说明

所有默认路径、随机种子和主要超参数集中定义在 `code/config.py`。评价图默认从 `output/main_results.json`、`output/conditional_results_with_wasserstein.json` 和 `output/robustness_extended_results.json` 读取指标，因此重新绘图不需要重新训练模型。
