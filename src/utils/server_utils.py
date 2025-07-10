"""
服务器环境工具函数
检测和配置服务器环境
"""

import os
import logging

logger = logging.getLogger(__name__)


def is_server_environment() -> bool:
    """
    检测是否为服务器环境
    
    Returns:
        是否为服务器环境
    """
    # 检查是否有显示环境
    has_display = 'DISPLAY' in os.environ
    
    # 检查是否在SSH会话中
    is_ssh = 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ
    
    # 检查是否在容器中
    is_container = os.path.exists('/.dockerenv') or 'container' in os.environ.get('HOSTNAME', '')
    
    # 检查是否在CI/CD环境中
    is_ci = any(key in os.environ for key in ['CI', 'CONTINUOUS_INTEGRATION', 'GITHUB_ACTIONS'])
    
    server_indicators = [not has_display, is_ssh, is_container, is_ci]
    
    if any(server_indicators):
        logger.info(f"Server environment detected: display={has_display}, ssh={is_ssh}, "
                   f"container={is_container}, ci={is_ci}")
        return True
    else:
        logger.info("Desktop environment detected")
        return False


def setup_server_matplotlib():
    """设置matplotlib为服务器兼容模式"""
    try:
        import matplotlib
        
        # 如果是服务器环境，设置为非交互式后端
        if is_server_environment():
            matplotlib.use('Agg')
            logger.info("Set matplotlib backend to 'Agg' for server environment")
        
        # 设置字体以避免中文显示问题
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    except ImportError:
        logger.warning("matplotlib not available")
    except Exception as e:
        logger.warning(f"Failed to setup matplotlib: {e}")


def setup_server_environment():
    """设置服务器环境变量和配置"""
    logger.info("Setting up server environment...")
    
    # 设置matplotlib后端
    os.environ['MPLBACKEND'] = 'Agg'
    
    # 禁用wandb交互式功能
    os.environ['WANDB_DISABLED'] = 'true'
    
    # 设置tokenizers并行处理
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # 设置CUDA相关
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 设置缓存目录
    cache_dir = os.path.join(os.getcwd(), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = os.path.join(cache_dir, 'huggingface')
    
    if 'TRANSFORMERS_CACHE' not in os.environ:
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
    
    # 设置线程数
    if 'OMP_NUM_THREADS' not in os.environ:
        import multiprocessing
        num_cores = multiprocessing.cpu_count()
        thread_count = min(4, num_cores)
        os.environ['OMP_NUM_THREADS'] = str(thread_count)
        os.environ['MKL_NUM_THREADS'] = str(thread_count)
        logger.info(f"Set thread count to {thread_count}")
    
    # 设置编码
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    logger.info("Server environment setup completed")


def get_server_config_adjustments() -> dict:
    """
    获取服务器环境的配置调整建议
    
    Returns:
        配置调整字典
    """
    adjustments = {}
    
    if is_server_environment():
        # 服务器环境的配置调整
        adjustments.update({
            'batch_size_multiplier': 0.75,  # 减少批处理大小
            'num_workers_max': 2,           # 限制工作进程数
            'enable_mixed_precision': True,  # 启用混合精度
            'gradient_checkpointing': True,  # 启用梯度检查点
            'dataloader_pin_memory': False,  # 禁用pin_memory
            'matplotlib_backend': 'Agg',     # 非交互式后端
            'disable_wandb': True,           # 禁用wandb
            'cache_cleanup': True            # 启用缓存清理
        })
    else:
        # 桌面环境的配置
        adjustments.update({
            'batch_size_multiplier': 1.0,
            'num_workers_max': 4,
            'enable_mixed_precision': False,
            'gradient_checkpointing': False,
            'dataloader_pin_memory': True,
            'matplotlib_backend': 'default',
            'disable_wandb': False,
            'cache_cleanup': False
        })
    
    return adjustments


def apply_server_optimizations(config: dict) -> dict:
    """
    应用服务器环境优化
    
    Args:
        config: 原始配置
        
    Returns:
        优化后的配置
    """
    adjustments = get_server_config_adjustments()
    optimized_config = config.copy()
    
    # 应用批处理大小调整
    if 'batch_size' in optimized_config:
        original_batch_size = optimized_config['batch_size']
        optimized_config['batch_size'] = max(1, int(original_batch_size * adjustments['batch_size_multiplier']))
        
        if optimized_config['batch_size'] != original_batch_size:
            logger.info(f"Adjusted batch_size: {original_batch_size} -> {optimized_config['batch_size']}")
    
    # 应用工作进程数调整
    if 'num_workers' in optimized_config:
        optimized_config['num_workers'] = min(optimized_config['num_workers'], adjustments['num_workers_max'])
    
    # 应用其他优化
    if adjustments['dataloader_pin_memory'] is False:
        optimized_config['pin_memory'] = False
    
    if adjustments['enable_mixed_precision']:
        optimized_config['mixed_precision'] = True
    
    logger.info("Applied server environment optimizations")
    return optimized_config


def check_server_resources():
    """检查服务器资源"""
    logger.info("Checking server resources...")
    
    # 检查CPU
    try:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        logger.info(f"CPU cores: {cpu_count}")
    except Exception as e:
        logger.warning(f"Could not check CPU: {e}")
    
    # 检查内存
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"Memory: {memory.total / 1024**3:.1f} GB total, "
                   f"{memory.available / 1024**3:.1f} GB available")
        
        if memory.available / 1024**3 < 4:
            logger.warning("Low memory available (< 4GB)")
            
    except ImportError:
        logger.warning("psutil not available, cannot check memory")
    
    # 检查磁盘空间
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        logger.info(f"Disk space: {free / 1024**3:.1f} GB free")
        
        if free / 1024**3 < 5:
            logger.warning("Low disk space (< 5GB)")
            
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name}, {gpu_memory:.1f} GB")
        else:
            logger.info("No GPU available")
    except ImportError:
        logger.warning("PyTorch not available, cannot check GPU")
