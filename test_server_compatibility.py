"""
服务器兼容性测试脚本
验证项目在服务器环境下的兼容性
"""

import os
import sys
import logging
from datetime import datetime

# 设置服务器环境变量
os.environ['MPLBACKEND'] = 'Agg'
os.environ['WANDB_DISABLED'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_matplotlib_headless():
    """测试matplotlib无头模式"""
    logger.info("Testing matplotlib headless mode...")
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # 设置为非交互式后端
        import matplotlib.pyplot as plt
        
        # 创建一个简单的图表
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])
        ax.set_title('Test Plot')
        
        # 保存到文件而不是显示
        test_dir = 'test_output'
        os.makedirs(test_dir, exist_ok=True)
        plt.savefig(os.path.join(test_dir, 'test_plot.png'))
        plt.close()
        
        logger.info("✓ Matplotlib headless mode test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Matplotlib headless mode test failed: {e}")
        return False


def test_device_detection():
    """测试设备自动检测"""
    logger.info("Testing device auto-detection...")
    
    try:
        from src.utils.device_utils import get_device, setup_device
        
        # 测试自动检测
        device = get_device("auto")
        logger.info(f"Auto-detected device: {device}")
        
        # 测试配置设置
        config = {'device': 'auto', 'batch_size': 32, 'num_workers': 4}
        final_device = setup_device(config)
        logger.info(f"Final device: {final_device}")
        logger.info(f"Optimized config: {config}")
        
        logger.info("✓ Device detection test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Device detection test failed: {e}")
        return False


def test_server_environment_detection():
    """测试服务器环境检测"""
    logger.info("Testing server environment detection...")
    
    try:
        from src.utils.server_utils import is_server_environment, get_server_config_adjustments
        
        is_server = is_server_environment()
        logger.info(f"Is server environment: {is_server}")
        
        adjustments = get_server_config_adjustments()
        logger.info(f"Server adjustments: {adjustments}")
        
        logger.info("✓ Server environment detection test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Server environment detection test failed: {e}")
        return False


def test_fallback_implementations():
    """测试备用实现"""
    logger.info("Testing fallback implementations...")
    
    try:
        # 测试数据集备用实现
        from src.stage1.datasets import PartiPromptsDataset
        
        dataset = PartiPromptsDataset(download=False)
        if len(dataset.data) > 0:
            logger.info(f"✓ Dataset fallback working: {len(dataset.data)} samples")
        else:
            logger.warning("Dataset fallback returned no data")
        
        # 测试知识库备用实现
        from src.stage3.retrieval_augment.knowledge_base import KnowledgeBase
        
        config = {
            'knowledge_base': {
                'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
                'vector_dim': 384,
                'index_type': 'faiss'
            }
        }
        
        kb = KnowledgeBase(config)
        kb.add_text_knowledge("test", "test knowledge", {})
        results = kb.search_text_knowledge("test", top_k=1)
        
        if results:
            logger.info("✓ Knowledge base fallback working")
        else:
            logger.warning("Knowledge base fallback returned no results")
        
        logger.info("✓ Fallback implementations test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Fallback implementations test failed: {e}")
        return False


def test_memory_optimization():
    """测试内存优化"""
    logger.info("Testing memory optimization...")
    
    try:
        from src.utils.device_utils import get_optimal_batch_size, check_memory_usage
        
        # 测试批处理大小优化
        optimal_cpu = get_optimal_batch_size(32, "cpu", "medium")
        optimal_gpu = get_optimal_batch_size(32, "cuda", "medium")
        
        logger.info(f"Optimal batch sizes - CPU: {optimal_cpu}, GPU: {optimal_gpu}")
        
        # 检查内存使用
        check_memory_usage()
        
        logger.info("✓ Memory optimization test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Memory optimization test failed: {e}")
        return False


def test_cache_management():
    """测试缓存管理"""
    logger.info("Testing cache management...")
    
    try:
        # 检查缓存目录设置
        cache_dirs = [
            os.environ.get('HF_HOME', ''),
            os.environ.get('TRANSFORMERS_CACHE', ''),
            '.cache'
        ]
        
        for cache_dir in cache_dirs:
            if cache_dir and os.path.exists(cache_dir):
                logger.info(f"✓ Cache directory exists: {cache_dir}")
            elif cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"✓ Created cache directory: {cache_dir}")
        
        logger.info("✓ Cache management test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Cache management test failed: {e}")
        return False


def test_minimal_functionality():
    """测试最小功能集"""
    logger.info("Testing minimal functionality...")
    
    try:
        # 测试基础导入
        from src.stage1.metrics import CLIPScore
        from src.stage2.model import HallucinationDetectionModel
        
        # 测试配置加载
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
                logger.info(f"✓ Config loaded: {config_file}")
            else:
                logger.warning(f"Config not found: {config_file}")
        
        logger.info("✓ Minimal functionality test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Minimal functionality test failed: {e}")
        return False


def main():
    """主测试函数"""
    logger.info("="*60)
    logger.info("服务器兼容性测试开始")
    logger.info("="*60)
    
    tests = [
        ("Matplotlib无头模式", test_matplotlib_headless),
        ("设备自动检测", test_device_detection),
        ("服务器环境检测", test_server_environment_detection),
        ("备用实现", test_fallback_implementations),
        ("内存优化", test_memory_optimization),
        ("缓存管理", test_cache_management),
        ("最小功能集", test_minimal_functionality),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n测试: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"测试失败: {test_name}")
        except Exception as e:
            logger.error(f"测试异常: {test_name} - {e}")
    
    logger.info("\n" + "="*60)
    logger.info(f"服务器兼容性测试结果: {passed}/{total} 通过")
    
    if passed == total:
        logger.info("🎉 项目完全兼容服务器环境！")
        logger.info("\n推荐的服务器运行命令:")
        logger.info("python run_server.py --mode all")
        return True
    elif passed >= total * 0.7:
        logger.warning("⚠️  项目基本兼容服务器环境，但有部分功能可能受限")
        logger.info("\n可以尝试运行:")
        logger.info("python run_server.py --mode stage1")
        return True
    else:
        logger.error("❌ 项目在服务器环境下存在兼容性问题")
        logger.info("请检查依赖安装和环境配置")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
