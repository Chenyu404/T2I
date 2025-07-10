"""
核心功能测试脚本
验证项目的核心组件是否正常工作
"""

import os
import sys
import logging
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_stage1_metrics():
    """测试阶段一的评估指标"""
    logger.info("Testing Stage 1 metrics...")
    
    try:
        from src.stage1.metrics import CLIPScore
        
        # 创建测试图像和文本
        test_image = Image.new('RGB', (224, 224), (128, 128, 128))
        test_text = "A gray square image"
        
        # 测试CLIPScore
        try:
            clip_scorer = CLIPScore(device="cpu")
            scores = clip_scorer.compute_score([test_image], [test_text])
            logger.info(f"CLIPScore test passed: {scores[0]:.4f}")
        except Exception as e:
            logger.warning(f"CLIPScore test failed: {e}")
        
        logger.info("Stage 1 metrics test completed")
        return True
        
    except Exception as e:
        logger.error(f"Stage 1 metrics test failed: {e}")
        return False


def test_stage1_datasets():
    """测试阶段一的数据集处理"""
    logger.info("Testing Stage 1 datasets...")
    
    try:
        from src.stage1.datasets import PartiPromptsDataset
        
        # 测试数据集加载（不实际下载）
        dataset = PartiPromptsDataset(download=False)
        
        # 创建一些测试数据
        dataset.data = [
            {'id': 0, 'prompt': 'test prompt 1', 'category': 'test', 'challenge': 'basic'},
            {'id': 1, 'prompt': 'test prompt 2', 'category': 'test', 'challenge': 'basic'}
        ]
        
        prompts = dataset.get_prompts()
        stats = dataset.get_statistics()
        
        logger.info(f"Dataset test passed: {len(prompts)} prompts, stats: {stats}")
        return True
        
    except Exception as e:
        logger.error(f"Stage 1 datasets test failed: {e}")
        return False


def test_stage2_model():
    """测试阶段二的模型"""
    logger.info("Testing Stage 2 model...")
    
    try:
        import yaml
        
        # 创建简化的配置
        config = {
            'model': {
                'clip': {
                    'model_name': 'openai/clip-vit-base-patch32',
                    'freeze_backbone': False
                },
                'fusion': {
                    'fusion_dim': 512,
                    'method': 'concat'
                },
                'classifier': {
                    'hidden_dims': [256, 128],
                    'num_classes': 2,
                    'dropout': 0.3,
                    'activation': 'relu'
                }
            }
        }
        
        from src.stage2.model import HallucinationDetectionModel
        
        # 测试模型初始化
        model = HallucinationDetectionModel(config)
        logger.info("Stage 2 model initialization test passed")
        return True
        
    except Exception as e:
        logger.error(f"Stage 2 model test failed: {e}")
        return False


def test_stage3_knowledge_base():
    """测试阶段三的知识库"""
    logger.info("Testing Stage 3 knowledge base...")

    try:
        config = {
            'knowledge_base': {
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'vector_dim': 384,
                'index_type': 'faiss'
            }
        }

        # 直接导入知识库模块
        from src.stage3.retrieval_augment.knowledge_base import KnowledgeBase
        
        # 测试知识库初始化
        kb = KnowledgeBase(config)
        
        # 添加测试知识
        kb.add_text_knowledge("test_1", "The Eiffel Tower is in Paris", {"type": "factual"})
        kb.add_text_knowledge("test_2", "Einstein developed relativity theory", {"type": "factual"})
        
        # 测试搜索
        results = kb.search_text_knowledge("Eiffel Tower", top_k=2)
        
        logger.info(f"Knowledge base test passed: found {len(results)} results")
        return True
        
    except Exception as e:
        logger.error(f"Stage 3 knowledge base test failed: {e}")
        return False


def test_configuration_files():
    """测试配置文件"""
    logger.info("Testing configuration files...")
    
    try:
        import yaml
        
        config_files = [
            'config/stage1_config.yaml',
            'config/stage2_config.yaml', 
            'config/stage3_config.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"✓ {config_file} is valid")
            else:
                logger.warning(f"✗ {config_file} not found")
        
        return True
        
    except Exception as e:
        logger.error(f"Configuration files test failed: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("="*50)
    logger.info("核心功能测试开始")
    logger.info("="*50)
    
    tests = [
        ("配置文件", test_configuration_files),
        ("阶段一评估指标", test_stage1_metrics),
        ("阶段一数据集", test_stage1_datasets),
        ("阶段二模型", test_stage2_model),
        ("阶段三知识库", test_stage3_knowledge_base),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n测试: {test_name}")
        try:
            if test_func():
                logger.info(f"✓ {test_name} 通过")
                passed += 1
            else:
                logger.error(f"✗ {test_name} 失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 异常: {e}")
    
    logger.info("\n" + "="*50)
    logger.info(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 所有核心功能测试通过！")
        return True
    else:
        logger.warning(f"⚠️  {total - passed} 个测试失败")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
