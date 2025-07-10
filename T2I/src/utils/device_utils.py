"""
设备工具函数
自动检测和配置最佳计算设备
"""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(device_config: str = "auto") -> str:
    """
    获取最佳计算设备
    
    Args:
        device_config: 设备配置 ("auto", "cuda", "cpu")
        
    Returns:
        设备字符串
    """
    if device_config == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            logger.info(f"Auto-detected CUDA device: {gpu_name}")
            logger.info(f"GPU count: {gpu_count}, Memory: {gpu_memory:.1f} GB")
            
            # 检查GPU内存是否足够
            if gpu_memory < 4.0:
                logger.warning("GPU memory < 4GB, consider using CPU for large models")
            
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
            
    elif device_config == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("Using CUDA as requested")
        else:
            device = "cpu"
            logger.warning("CUDA requested but not available, falling back to CPU")
            
    elif device_config == "cpu":
        device = "cpu"
        logger.info("Using CPU as requested")
        
    else:
        logger.warning(f"Unknown device config: {device_config}, using auto")
        return get_device("auto")
    
    return device


def setup_device(config: dict) -> str:
    """
    根据配置设置设备
    
    Args:
        config: 配置字典
        
    Returns:
        设备字符串
    """
    device_config = config.get('device', 'auto')
    device = get_device(device_config)
    
    # 更新配置中的设备设置
    config['device'] = device
    
    # 根据设备调整其他配置
    if device == "cpu":
        # CPU环境下的优化
        if 'batch_size' in config:
            original_batch_size = config['batch_size']
            config['batch_size'] = max(1, original_batch_size // 2)
            logger.info(f"Reduced batch_size for CPU: {original_batch_size} -> {config['batch_size']}")
        
        if 'num_workers' in config:
            config['num_workers'] = min(config['num_workers'], 2)
            logger.info(f"Reduced num_workers for CPU: {config['num_workers']}")
    
    elif device == "cuda":
        # GPU环境下的优化
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory < 8.0 and 'batch_size' in config:
                # 低显存GPU的优化
                original_batch_size = config['batch_size']
                config['batch_size'] = max(1, original_batch_size // 2)
                logger.info(f"Reduced batch_size for low GPU memory: {original_batch_size} -> {config['batch_size']}")
                
        except Exception as e:
            logger.warning(f"Could not check GPU memory: {e}")
    
    return device


def get_optimal_batch_size(base_batch_size: int, device: str, model_size: str = "medium") -> int:
    """
    获取最优批处理大小
    
    Args:
        base_batch_size: 基础批处理大小
        device: 计算设备
        model_size: 模型大小 ("small", "medium", "large")
        
    Returns:
        优化后的批处理大小
    """
    if device == "cpu":
        # CPU环境下减少批处理大小
        multiplier = 0.5
    elif device == "cuda":
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            if gpu_memory >= 16:
                multiplier = 1.5  # 高端GPU可以增加批处理大小
            elif gpu_memory >= 8:
                multiplier = 1.0  # 中端GPU保持原始大小
            else:
                multiplier = 0.5  # 低端GPU减少批处理大小
                
        except Exception:
            multiplier = 0.75  # 默认稍微减少
    else:
        multiplier = 1.0
    
    # 根据模型大小调整
    size_multipliers = {
        "small": 1.5,
        "medium": 1.0,
        "large": 0.5
    }
    
    multiplier *= size_multipliers.get(model_size, 1.0)
    
    optimal_batch_size = max(1, int(base_batch_size * multiplier))
    
    if optimal_batch_size != base_batch_size:
        logger.info(f"Adjusted batch size: {base_batch_size} -> {optimal_batch_size} "
                   f"(device: {device}, model: {model_size})")
    
    return optimal_batch_size


def check_memory_usage():
    """检查内存使用情况"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        logger.info(f"System memory: {memory.total / 1024**3:.1f} GB total, "
                   f"{memory.available / 1024**3:.1f} GB available")
        
        if memory.percent > 80:
            logger.warning("High system memory usage detected")
            
    except ImportError:
        logger.debug("psutil not available, cannot check system memory")
    
    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            
            logger.info(f"GPU memory: {gpu_memory:.1f} GB total, "
                       f"{allocated:.1f} GB allocated, {cached:.1f} GB cached")
            
            if allocated / gpu_memory > 0.8:
                logger.warning("High GPU memory usage detected")
                
        except Exception as e:
            logger.debug(f"Could not check GPU memory: {e}")


def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")
