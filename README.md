# 文本生成图片幻觉检测与消除项目

## 项目概述

本项目旨在探索如何识别并缓解"文生图"任务中的幻觉问题（Hallucination）。该问题主要指人工智能模型生成图像在语义或细节上与输入文本不一致，或图像内容违背基本事实知识。

## 项目目标

掌握文本生成图像质量评估方法、幻觉类型的分类标准，以及幻觉检测与消除的基本算法。

## 项目结构

```
fuwut/
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

## 🔧 数据集兼容性


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

### 备用机制
- 所有数据集都有fallback实现，无网络时可正常运行
- 自动检测数据集可用性并切换到备用模式
- 详细信息请参考 [数据集兼容性修复报告](DATASET_COMPATIBILITY_FIXES.md)

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

# 或者分步骤运行
python run_project.py --mode setup      # 环境设置
python run_project.py --mode download   # 下载数据集
python run_project.py --mode stage1     # 运行阶段一
python run_project.py --mode stage2     # 运行阶段二
python run_project.py --mode stage3     # 运行阶段三
```

### 服务器环境 (无GUI)
```bash
# 服务器兼容性测试
python test_server_compatibility.py

# 服务器专用运行脚本
python run_server.py --mode all

# 或分阶段运行
python run_server.py --mode stage1
python run_server.py --mode stage2
python run_server.py --mode stage3
```

### 详细安装与使用

1. **环境设置**：
```bash
# 检查环境并安装依赖
python scripts/setup_environment.py

# 或手动安装
pip install -r requirements.txt
```

2. **下载数据集**：
```bash
# 下载所有数据集
python scripts/download_datasets.py --datasets all

# 或下载特定数据集
python scripts/download_datasets.py --datasets parti_prompts t2i_compbench t2i_factualbench
```

3. **运行各阶段**：

**阶段一：文生图评估系统**
```bash
# 基本运行
python scripts/run_stage1.py

# 自定义参数
python scripts/run_stage1.py --max-samples 100 --generate-images
```

**阶段二：幻觉检测模型训练**
```bash
# 训练和评估
python scripts/run_stage2.py both --generate-plots

# 仅训练
python scripts/run_stage2.py train

# 仅评估
python scripts/run_stage2.py eval --model-path models/stage2/best_model.pth
```

**阶段三：创新幻觉缓解方案**
```bash
# 运行所有组件
python scripts/run_stage3.py all

# 运行特定组件
python scripts/run_stage3.py build_kb      # 构建知识库
python scripts/run_stage3.py train_rl      # 训练RL智能体
python scripts/run_stage3.py test_retrieval # 测试检索系统
python scripts/run_stage3.py demo          # 运行演示
```

## 输出结果

项目运行完成后，将在以下目录生成结果：

- `results/stage1/` - 阶段一评估结果和可视化
- `results/stage2/` - 阶段二训练结果和模型性能分析
- `results/stage3/` - 阶段三创新方案演示结果
- `models/` - 训练好的模型文件
- `logs/` - 详细的运行日志
- `results/final_report.md` - 项目最终报告

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

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减少批处理大小
   # 在配置文件中修改 batch_size 参数
   ```

2. **数据集下载失败**
   ```bash
   # 检查网络连接，或使用代理
   # 手动下载数据集到 data/ 目录
   ```

3. **模型加载错误**
   ```bash
   # 检查模型文件路径
   # 重新训练模型
   python scripts/run_stage2.py train
   ```

### 系统要求

**桌面环境:**
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (推荐，用于GPU加速)
- 内存: 16GB+ (推荐)
- 存储: 10GB+ 可用空间

**服务器环境:**
- Linux/Unix系统 (Ubuntu 18.04+, CentOS 7+)
- Python 3.8+
- 内存: 8GB+ (最低), 16GB+ (推荐)
- 存储: 10GB+ 可用空间
- 网络: 能访问Hugging Face Hub
- 详细要求请参考 [服务器部署指南](SERVER_DEPLOYMENT_GUIDE.md)

## 参考文献

[1] Hu Y, Liu B, Kasai J, et al. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 20406-20417.

[2] Liang Y, He J, Li G, et al. Rich human feedback for text-to-image generation[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 19401-19411.

[3] Huang Z, He W, Long Q, et al. T2I-FactualBench: Benchmarking the Factuality of Text-to-Image Models with Knowledge-Intensive Concepts[J]. ACL, 2025.

[4] Han S, Fan H, Fu J, et al. EvalMuse-40K: A Reliable and Fine-Grained Benchmark with Comprehensive Human Annotations for Text-to-Image Generation Model Evaluation[J]. arXiv preprint arXiv:2412.18150, 2024.

## 许可证

本项目仅用于学术研究目的。

## 联系方式

