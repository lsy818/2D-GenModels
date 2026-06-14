# Code README

本目录包含二维生成模型实验的全部可执行代码。建议从仓库根目录运行命令，保证路径统一。

## 环境配置

推荐 Python 3.10+。安装依赖：

```bash
pip install -r requirements.txt
```

## 脚本结构

```text
code/
├── run.py                         # 主入口：训练、扩展实验与图表生成
├── config.py                      # 全局配置：路径、类别名、随机种子与超参数
├── generate_data.py               # 数据生成：四类二维分布与隐藏测试集
├── data.py                        # 数据读取与几何统计工具
├── metrics.py                     # 评价指标：MMD、Wasserstein、Coverage、Precision
├── experiments.py                 # 主流程扩展：条件生成与鲁棒性实验
├── visualization.py               # 图表生成：读取 output/*.json 并输出 PDF
├── models/                        # 模型定义
│   ├── base.py                    # 生成模型基类
│   ├── vae_model.py               # VAE / beta-VAE 实现
│   └── diffusion_model.py         # DDPM 实现
└── extensions/                    # 诊断、调参与结构增强实验
    ├── __init__.py                # 扩展实验包初始化
    ├── vae_diagnostics.py         # VAE 潜空间与后验诊断
    ├── vae_improvement_experiments.py # VAE 改进实验
    ├── vae_beta_sweep.py          # VAE beta sweep 实验
    ├── diffusion_spiral_experiments.py # Spiral DDPM 变体
    └── spiral_structure_enhancement.py # Spiral 结构增强
```

## 常用命令

生成数据：

```bash
python -m code.generate_data
```

训练主模型、运行扩展实验并生成图：

```bash
python -m code.run
```

只重新生成图：

```bash
python -m code.run --figs-only
```

只运行 `experiments.py` 中的扩展实验：

```bash
python -m code.run --ext-only
```

运行分析与增强实验：

```bash
python -m code.extensions.vae_diagnostics
python -m code.extensions.vae_improvement_experiments
python -m code.extensions.vae_beta_sweep
python -m code.extensions.diffusion_spiral_experiments
python -m code.extensions.spiral_structure_enhancement
```

## 数据生成

默认命令会为四类分布分别生成 2000 个训练样本、2000 个测试样本和 2000 个隐藏测试样本，并写入仓库根目录下的 `data/`：

```bash
python -m code.generate_data
```

输出文件：

```text
data/
├── train.npy              # 训练样本，shape = (8000, 2)
├── test.npy               # 测试样本，shape = (8000, 2)
├── train_label.npy        # 训练标签
├── test_label.npy         # 测试标签
├── hidden_test.npy        # 隐藏测试样本
├── hidden_test_label.npy  # 隐藏测试标签
└── metadata.json          # 数据规模、类别名和随机种子
```

标签含义：

```text
0: Gaussian Mixture
1: Ring
2: Two Moons
3: Spiral
```

常用参数：

```bash
python -m code.generate_data --seed 20260525
python -m code.generate_data --train-per-class 2000 --test-per-class 2000 --hidden-per-class 2000
python -m code.generate_data --output-dir data
```

如果环境安装了 `matplotlib`，可以额外生成数据预览 PNG：

```bash
python -m code.generate_data --plot
```

## 输出位置

```text
data/                数据集与标签
models/              主模型权重
models/extensions/   扩展实验模型权重
figures/analysis/    数据几何分析图
figures/generation/  生成样本与过程图
figures/evaluation/  指标、条件生成、鲁棒性图
figures/extensions/  诊断、调参和结构增强图
output/              实验数值结果 JSON
```

## 说明

- 所有默认路径定义在 `config.py` 中。
- `generate_data.py` 默认写入仓库根目录下的 `data/`；`--plot` 会额外生成数据预览 PNG。
- `visualization.py` 默认只保存 PDF 图，评价类图表读取 `output/main_results.json`、`output/conditional_results_with_wasserstein.json` 和 `output/robustness_extended_results.json`。
- 扩展脚本统一放在 `code/extensions/` 下，便于和 `figures/extensions/`、`models/extensions/` 对齐。
- 随机种子在 `config.py` 和各实验脚本中固定，便于复现实验。
