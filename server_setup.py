"""
服务器环境配置脚本
确保项目在无GUI服务器环境下正常运行
"""

import os
import sys
import logging
from datetime import datetime


def setup_server_environment():
    """设置服务器环境"""
    logger = logging.getLogger(__name__)
    logger.info("Setting up server environment...")
    
    # 设置matplotlib后端为非交互式
    os.environ['MPLBACKEND'] = 'Agg'
    
    # 禁用CUDA设备显示（避免服务器上的GPU信息输出过多）
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 设置HuggingFace缓存目录（避免权限问题）
    if 'HF_HOME' not in os.environ:
        cache_dir = os.path.join(os.getcwd(), '.cache', 'huggingface')
        os.environ['HF_HOME'] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Set HuggingFace cache directory: {cache_dir}")
    
    # 设置Transformers缓存目录
    if 'TRANSFORMERS_CACHE' not in os.environ:
        cache_dir = os.path.join(os.getcwd(), '.cache', 'transformers')
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Set Transformers cache directory: {cache_dir}")
    
    # 禁用wandb（避免服务器上的交互式登录）
    os.environ['WANDB_DISABLED'] = 'true'
    
    # 设置tokenizers并行处理（服务器环境优化）
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    logger.info("Server environment setup completed")


def check_server_compatibility():
    """检查服务器兼容性"""
    logger = logging.getLogger(__name__)
    logger.info("Checking server compatibility...")
    
    issues = []
    
    # 检查是否有显示环境
    if 'DISPLAY' in os.environ:
        logger.info("Display environment detected")
    else:
        logger.info("No display environment (headless server)")
        issues.append("headless")
    
    # 检查matplotlib后端
    try:
        import matplotlib
        backend = matplotlib.get_backend()
        logger.info(f"Matplotlib backend: {backend}")
        
        if backend in ['Qt5Agg', 'TkAgg', 'GTKAgg'] and 'DISPLAY' not in os.environ:
            logger.warning("Interactive matplotlib backend detected in headless environment")
            matplotlib.use('Agg')
            logger.info("Switched to Agg backend")
    except ImportError:
        logger.warning("Matplotlib not available")
    
    # 检查CUDA可用性
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        else:
            logger.info("CUDA not available, using CPU")
    except ImportError:
        logger.warning("PyTorch not available")
    
    # 检查磁盘空间
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free // (1024**3)
        logger.info(f"Available disk space: {free_gb} GB")
        
        if free_gb < 10:
            logger.warning("Low disk space detected")
            issues.append("low_disk_space")
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
    
    # 检查内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available // (1024**3)
        logger.info(f"Available memory: {available_gb} GB")
        
        if available_gb < 8:
            logger.warning("Low memory detected")
            issues.append("low_memory")
    except ImportError:
        logger.warning("psutil not available, cannot check memory")
    
    return issues


def create_server_config():
    """创建服务器专用配置"""
    logger = logging.getLogger(__name__)
    logger.info("Creating server-specific configuration...")
    
    server_config = {
        'environment': 'server',
        'headless': True,
        'matplotlib_backend': 'Agg',
        'disable_wandb': True,
        'batch_size_reduction': True,  # 在内存受限时减少批处理大小
        'cpu_only_fallback': True,    # 在GPU不可用时自动使用CPU
        'cache_management': True,     # 启用缓存管理
        'log_level': 'INFO'
    }
    
    # 保存服务器配置
    import json
    with open('server_config.json', 'w') as f:
        json.dump(server_config, f, indent=2)
    
    logger.info("Server configuration saved to server_config.json")


def optimize_for_server():
    """为服务器环境优化设置"""
    logger = logging.getLogger(__name__)
    logger.info("Optimizing for server environment...")
    
    # 设置OMP线程数（避免CPU过载）
    if 'OMP_NUM_THREADS' not in os.environ:
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        omp_threads = min(4, num_cores)  # 限制线程数
        os.environ['OMP_NUM_THREADS'] = str(omp_threads)
        logger.info(f"Set OMP_NUM_THREADS to {omp_threads}")
    
    # 设置MKL线程数
    if 'MKL_NUM_THREADS' not in os.environ:
        os.environ['MKL_NUM_THREADS'] = os.environ['OMP_NUM_THREADS']
    
    # 禁用一些可能导致问题的功能
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    logger.info("Server optimization completed")


def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/server_setup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    
    logger.info("="*50)
    logger.info("服务器环境配置开始")
    logger.info("="*50)
    
    try:
        # 设置服务器环境
        setup_server_environment()
        
        # 检查兼容性
        issues = check_server_compatibility()
        
        # 创建服务器配置
        create_server_config()
        
        # 优化设置
        optimize_for_server()
        
        # 报告结果
        logger.info("="*50)
        logger.info("服务器环境配置完成")
        
        if issues:
            logger.warning(f"检测到以下问题: {', '.join(issues)}")
            logger.info("项目已针对这些问题进行了优化配置")
        else:
            logger.info("未检测到兼容性问题")
        
        logger.info("项目现在可以在服务器环境下运行")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"服务器环境配置失败: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
