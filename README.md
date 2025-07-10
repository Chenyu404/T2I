# 文本生成图片幻觉检测与消除项目

## 项目概述

本项目旨在探索如何识别并缓解"文生图"任务中的幻觉问题（Hallucination）。主要指人工智能模型生成图像在语义或细节上与输入文本不一致，或图像内容违背基本事实知识。

## 项目目标

掌握文本生成图像质量评估方法、幻觉类型的分类标准，以及幻觉检测与消除的基本算法。

## 项目结构

```
T2I/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包列表
├── config/                      # 配置文件
│   ├── stage1_config.yaml      # 阶段一配置
│   ├── stage2_config.yaml      # 阶段二配置
│   └── stage3_config.yaml      # 阶段三配置
├── src/                         # 源代码
│   ├── __init__.py
│   ├── stage1/                  # 阶段一：评估系统
│   │   ├── __init__.py
│   │   ├── metrics/             # 评估指标实现
│   │   ├── datasets/            # 数据集处理
│   │   └── evaluation.py       # 评估脚本
│   ├── stage2/                  # 阶段二：幻觉检测
│   │   ├── __init__.py
│   │   ├── model.py            # 检测模型
│   │   ├── train.py            # 训练脚本
│   │   └── evaluate.py         # 评估脚本
│   └── stage3/                  # 阶段三：创新方案
│       ├── __init__.py
│       ├── rl_alignment/        # 强化学习对齐
│       └── retrieval_augment/   # 检索增强
├── data/                        # 数据目录
│   ├── parti_prompts/          # PartiPrompts数据集
│   ├── t2i_factualbench/       # T2I-FactualBench数据集
│   └── evalmuse/               # EvalMuse数据集
├── models/                      # 模型保存目录
├── results/                     # 结果输出目录
└── scripts/                     # 运行脚本
    ├── setup_environment.py    # 环境设置
    ├── download_datasets.py    # 数据集下载
    ├── run_stage1.py           # 运行阶段一
    ├── run_stage2.py           # 运行阶段二
    └── run_stage3.py           # 运行阶段三
```


### 验证数据集兼容性
```bash
# 运行数据集兼容性测试
python test_dataset_compatibility.py
```

### 使用的数据集
- **PartiPrompts**: `nateraw/parti-prompts` - 文生图提示词数据集
- **T2I-CompBench**: `NinaKarine/t2i-compbench` - 文生图组合评估基准
- **T2I-FactualBench**: `Sakeoffellow001/T2i_Factualbench` - 文生图事实性评估基准
- **EvalMuse**: `DY-Evalab/EvalMuse` - 文生图评估标注数据集


## 三个阶段说明

### 阶段一：文生图评估系统构建
- 实现6种评估指标：CLIPScore、ImageReward、PickScore、TIFA、FID、IS
- 处理数据集：PartiPrompts、T2I-CompBench、T2I-FactualBench
- 构建统一评估脚本

### 阶段二：幻觉检测模型训练
- 使用EvalMuse数据集训练幻觉检测模型
- 分析不同类型幻觉的检测效果

### 阶段三：创新幻觉缓解方案
- 基于强化学习的对齐方法（语义细节幻觉）
- 多模态检索增强机制（事实性幻觉）

## 快速开始

### 桌面环境 (有GUI)
```bash
# 运行完整项目（推荐）
python run_project.py --mode all

```

### 服务器环境 (无GUI)
```bash
# 服务器兼容性测试
python test_server_compatibility.py

# 服务器专用运行脚本
python run_server.py --mode all

```

### 详细安装与使用

1. **环境设置**：
```bash
# 检查环境并安装依赖
python scripts/setup_environment.py
```

2. **下载数据集**：
```bash
# 下载所有数据集
python scripts/download_datasets.py --datasets all
```

3. **运行各阶段**：

**阶段一：文生图评估系统**
```bash
# 基本运行
python scripts/run_stage1.py
```

**阶段二：幻觉检测模型训练**
```bash
# 训练和评估
python scripts/run_stage2.py both --generate-plots
```

**阶段三：创新幻觉缓解方案**
```bash
# 运行所有组件
python scripts/run_stage3.py all
```


## 技术特点

1. **多模态评估**: 实现6种不同的图文一致性评估指标
2. **深度学习检测**: 基于CLIP的多模态幻觉检测模型
3. **强化学习对齐**: 创新的RL方法用于幻觉定位和纠正
4. **知识增强**: 多模态检索增强机制提升事实一致性
5. **可解释性**: 提供幻觉区域定位和原因分析

## 创新点

1. **细粒度检测**: 不仅检测是否有幻觉，还能定位具体区域
2. **多类型覆盖**: 支持语义不一致、事实错误、对象幻觉等多种类型
3. **端到端流程**: 从检测到纠正的完整解决方案
4. **知识驱动**: 结合外部知识库提升纠正准确性



## 参考文献

[1] Hu Y, Liu B, Kasai J, et al. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 20406-20417.

[2] Liang Y, He J, Li G, et al. Rich human feedback for text-to-image generation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 19401-19411.

[3] Huang Z, He W, Long Q, et al. T2I-FactualBench: Benchmarking the Factuality of Text-to-Image Models with Knowledge-Intensive Concepts[J]. ACL, 2025.

[4] Han S, Fan H, Fu J, et al. EvalMuse-40K: A Reliable and Fine-Grained Benchmark with Comprehensive Human Annotations for Text-to-Image Generation Model Evaluation[J]. arXiv preprint arXiv:2412.18150, 2024.

## 许可证

本项目仅用于学术研究目的。

## 联系方式
19855822550@163.com

