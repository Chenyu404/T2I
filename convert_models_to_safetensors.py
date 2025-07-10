#!/usr/bin/env python3
"""
模型格式转换脚本
将现有的.pth模型文件转换为safetensors格式
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_model_file(pth_path: str) -> bool:
    """
    转换单个模型文件到safetensors格式
    
    Args:
        pth_path: .pth文件路径
        
    Returns:
        是否转换成功
    """
    try:
        # 检查依赖
        try:
            import torch
            from safetensors.torch import save_file as safetensors_save
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Please run: pip install torch>=2.6.0 safetensors>=0.4.0")
            return False
        
        logger.info(f"Converting {pth_path}...")
        
        # 加载原始模型
        try:
            # 尝试使用weights_only=True（PyTorch 2.6+）
            checkpoint = torch.load(pth_path, map_location='cpu', weights_only=True)
        except TypeError:
            # 回退到旧版本加载方式
            logger.warning("Using legacy torch.load (consider upgrading PyTorch)")
            checkpoint = torch.load(pth_path, map_location='cpu')
        
        # 准备safetensors路径
        safetensors_path = pth_path.replace('.pth', '.safetensors')
        metadata_path = pth_path.replace('.pth', '_metadata.json')
        
        # 分离模型权重和元数据
        if isinstance(checkpoint, dict):
            # 提取模型权重
            model_weights = {}
            metadata = {}
            
            for key, value in checkpoint.items():
                if key.endswith('_state_dict') and isinstance(value, dict):
                    # 这是模型权重
                    prefix = key.replace('_state_dict', '')
                    for param_name, param_value in value.items():
                        if hasattr(param_value, 'dtype'):  # 是tensor
                            model_weights[f"{prefix}.{param_name}"] = param_value
                        else:
                            metadata[f"{prefix}.{param_name}"] = param_value
                elif hasattr(value, 'dtype'):  # 直接的tensor
                    model_weights[key] = value
                else:
                    # 其他元数据
                    metadata[key] = value
            
            # 保存模型权重为safetensors
            if model_weights:
                safetensors_save(model_weights, safetensors_path)
                logger.info(f"Model weights saved to {safetensors_path}")
            
            # 保存元数据为JSON
            if metadata:
                # 确保元数据可以JSON序列化
                serializable_metadata = {}
                for key, value in metadata.items():
                    try:
                        json.dumps(value)
                        serializable_metadata[key] = value
                    except (TypeError, ValueError):
                        serializable_metadata[key] = str(value)
                
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable_metadata, f, ensure_ascii=False, indent=2)
                logger.info(f"Metadata saved to {metadata_path}")
        
        else:
            # 直接是state_dict
            safetensors_save(checkpoint, safetensors_path)
            logger.info(f"Model saved to {safetensors_path}")
        
        logger.info(f"✅ Successfully converted {pth_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to convert {pth_path}: {e}")
        return False


def find_model_files(directory: str) -> list:
    """查找目录中的所有.pth文件"""
    pth_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                pth_files.append(os.path.join(root, file))
    return pth_files


def main():
    """主函数"""
    logger.info("Starting model conversion to safetensors format...")
    
    # 查找所有模型文件
    model_dirs = ['models', 'results']
    all_pth_files = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            pth_files = find_model_files(model_dir)
            all_pth_files.extend(pth_files)
            logger.info(f"Found {len(pth_files)} .pth files in {model_dir}")
    
    if not all_pth_files:
        logger.info("No .pth files found to convert")
        return True
    
    # 转换所有文件
    success_count = 0
    for pth_file in all_pth_files:
        if convert_model_file(pth_file):
            success_count += 1
    
    logger.info(f"Conversion completed: {success_count}/{len(all_pth_files)} files converted successfully")
    
    if success_count == len(all_pth_files):
        logger.info("✅ All model files converted successfully!")
        logger.info("You can now safely use the updated code with safetensors support")
        return True
    else:
        logger.warning(f"⚠️  {len(all_pth_files) - success_count} files failed to convert")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
