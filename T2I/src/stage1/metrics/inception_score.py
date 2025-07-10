"""
Inception Score (IS)评估指标实现
基于Inception网络计算生成图像的质量和多样性
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class InceptionScore:
    """
    Inception Score评估指标
    计算生成图像的质量和多样性
    """
    
    def __init__(self, batch_size: int = 32, splits: int = 10, device: str = "cuda"):
        """
        初始化Inception Score评估器
        
        Args:
            batch_size: 批处理大小
            splits: 分割数量用于计算标准差
            device: 计算设备
        """
        self.batch_size = batch_size
        self.splits = splits
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info("Loading Inception v3 model for IS...")
        
        # 加载预训练的Inception v3模型
        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Inception Score initialized successfully")
    
    def get_predictions(self, images: List[Image.Image]) -> np.ndarray:
        """
        获取图像的预测概率
        
        Args:
            images: 图像列表
            
        Returns:
            预测概率数组
        """
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(images), self.batch_size):
                batch_images = images[i:i+self.batch_size]
                
                # 预处理图像
                batch_tensors = []
                for img in batch_images:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    tensor = self.transform(img)
                    batch_tensors.append(tensor)
                
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # 获取预测
                outputs = self.model(batch_tensor)
                probs = F.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def calculate_inception_score(self, predictions: np.ndarray) -> Tuple[float, float]:
        """
        计算Inception Score
        
        Args:
            predictions: 预测概率数组
            
        Returns:
            IS均值和标准差
        """
        # 计算边际分布
        marginal = np.mean(predictions, axis=0)
        
        # 计算每个分割的IS
        scores = []
        split_size = predictions.shape[0] // self.splits
        
        for i in range(self.splits):
            start_idx = i * split_size
            if i == self.splits - 1:
                end_idx = predictions.shape[0]
            else:
                end_idx = (i + 1) * split_size
            
            split_predictions = predictions[start_idx:end_idx]
            
            # 计算KL散度
            kl_divs = []
            for pred in split_predictions:
                # 避免log(0)
                pred = np.clip(pred, 1e-10, 1.0)
                marginal_clipped = np.clip(marginal, 1e-10, 1.0)
                
                kl_div = np.sum(pred * np.log(pred / marginal_clipped))
                kl_divs.append(kl_div)
            
            # 计算该分割的IS
            split_score = np.exp(np.mean(kl_divs))
            scores.append(split_score)
        
        return np.mean(scores), np.std(scores)
    
    def compute_inception_score(self, images: List[Image.Image]) -> Tuple[float, float]:
        """
        计算图像列表的Inception Score
        
        Args:
            images: 图像列表
            
        Returns:
            IS均值和标准差
        """
        logger.info(f"Computing Inception Score for {len(images)} images")
        
        # 获取预测
        predictions = self.get_predictions(images)
        
        # 计算IS
        is_mean, is_std = self.calculate_inception_score(predictions)
        
        return float(is_mean), float(is_std)
    
    def compute_score(self, images: List[Image.Image], texts: List[str] = None) -> List[float]:
        """
        计算IS分数（为了与其他指标保持一致的接口）
        
        Args:
            images: 图像列表
            texts: 文本列表（IS不需要文本，此参数被忽略）
            
        Returns:
            IS分数列表（每个图像返回相同的IS均值）
        """
        is_mean, is_std = self.compute_inception_score(images)
        
        # 返回每个图像相同的IS分数
        return [is_mean] * len(images)
    
    def compute_detailed_score(self, images: List[Image.Image]) -> dict:
        """
        计算详细的IS分数信息
        
        Args:
            images: 图像列表
            
        Returns:
            详细分数信息
        """
        is_mean, is_std = self.compute_inception_score(images)
        
        return {
            'inception_score_mean': is_mean,
            'inception_score_std': is_std,
            'num_images': len(images),
            'splits': self.splits
        }
    
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
            'count': len(scores),
            'is_value': scores[0] if scores else 0.0  # IS是单一值
        }
