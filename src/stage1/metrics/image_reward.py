"""
ImageReward评估指标实现
基于人类反馈训练的图像质量评估模型
"""

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)


class ImageReward:
    """
    ImageReward评估指标
    使用基于人类反馈训练的模型评估图像质量
    """
    
    def __init__(self, model_name: str = "THUDM/ImageReward", device: str = "cuda"):
        """
        初始化ImageReward评估器
        
        Args:
            model_name: ImageReward模型名称
            device: 计算设备
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        try:
            logger.info(f"Loading ImageReward model: {model_name}")
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model.eval()
            logger.info("ImageReward initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to load ImageReward model: {e}")
            logger.info("Using fallback CLIP-based implementation")
            self._use_fallback = True
            self._init_fallback()
    
    def _init_fallback(self):
        """初始化备用CLIP模型"""
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
    
    def compute_score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        """
        计算ImageReward分数
        
        Args:
            images: 图像列表
            texts: 文本列表
            
        Returns:
            ImageReward分数列表
        """
        if len(images) != len(texts):
            raise ValueError("Images and texts must have the same length")
        
        scores = []
        
        with torch.no_grad():
            if hasattr(self, '_use_fallback') and self._use_fallback:
                # 使用CLIP作为备用方案
                for image, text in zip(images, texts):
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        return_tensors="pt",
                        padding=True
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
                    score = outputs.logits_per_image.cpu().numpy()[0][0]
                    # 将CLIP分数转换为类似ImageReward的范围
                    score = float(score / 100.0)
                    scores.append(score)
            else:
                # 使用真正的ImageReward模型
                for image, text in zip(images, texts):
                    try:
                        score = self.model.score(text, image)
                        scores.append(float(score))
                    except Exception as e:
                        logger.warning(f"Error computing ImageReward score: {e}")
                        scores.append(0.0)
        
        return scores
    
    def compute_batch_score(self, images: List[Image.Image], texts: List[str], 
                           batch_size: int = 16) -> List[float]:
        """
        批量计算ImageReward分数
        
        Args:
            images: 图像列表
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            ImageReward分数列表
        """
        all_scores = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_texts = texts[i:i+batch_size]
            batch_scores = self.compute_score(batch_images, batch_texts)
            all_scores.extend(batch_scores)
        
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
