"""
CLIPScore评估指标实现
基于CLIP模型计算图像和文本之间的相似度分数
"""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Union, Tuple
import logging

logger = logging.getLogger(__name__)


class CLIPScore:
    """
    CLIPScore评估指标
    使用CLIP模型计算图像-文本相似度分数
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda"):
        """
        初始化CLIPScore评估器
        
        Args:
            model_name: CLIP模型名称
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        logger.info(f"Loading CLIP model: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        self.model.eval()
        logger.info("CLIPScore initialized successfully")
    
    def compute_score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        """
        计算图像-文本对的CLIPScore
        
        Args:
            images: 图像列表
            texts: 文本列表
            
        Returns:
            CLIPScore分数列表
        """
        if len(images) != len(texts):
            raise ValueError("Images and texts must have the same length")
        
        scores = []
        
        with torch.no_grad():
            for image, text in zip(images, texts):
                # 预处理图像和文本
                inputs = self.processor(
                    text=[text], 
                    images=[image], 
                    return_tensors="pt", 
                    padding=True
                ).to(self.device)
                
                # 获取特征
                outputs = self.model(**inputs)
                
                # 计算相似度分数
                logits_per_image = outputs.logits_per_image
                score = logits_per_image.cpu().numpy()[0][0]
                scores.append(float(score))
        
        return scores
    
    def compute_batch_score(self, images: List[Image.Image], texts: List[str], 
                           batch_size: int = 16) -> List[float]:
        """
        批量计算CLIPScore
        
        Args:
            images: 图像列表
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            CLIPScore分数列表
        """
        if len(images) != len(texts):
            raise ValueError("Images and texts must have the same length")
        
        all_scores = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            
            with torch.no_grad():
                # 预处理批次数据
                inputs = self.processor(
                    text=batch_texts,
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                # 获取特征
                outputs = self.model(**inputs)
                
                # 计算相似度分数
                logits_per_image = outputs.logits_per_image
                scores = torch.diag(logits_per_image).cpu().numpy()
                all_scores.extend(scores.tolist())
        
        return all_scores
    
    def compute_statistics(self, scores: List[float]) -> dict:
        """
        计算分数统计信息
        
        Args:
            scores: 分数列表
            
        Returns:
            统计信息字典
        """
        scores_array = np.array(scores)
        
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'count': len(scores)
        }
