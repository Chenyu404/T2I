"""
FID (Frechet Inception Distance)评估指标实现
计算生成图像与真实图像分布之间的距离
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from PIL import Image
import numpy as np
from scipy import linalg
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class FID:
    """
    FID评估指标
    计算Frechet Inception Distance
    """
    
    def __init__(self, batch_size: int = 50, dims: int = 2048, device: str = "cuda"):
        """
        初始化FID评估器
        
        Args:
            batch_size: 批处理大小
            dims: 特征维度
            device: 计算设备
        """
        self.batch_size = batch_size
        self.dims = dims
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info("Loading Inception v3 model for FID...")
        
        # 加载预训练的Inception v3模型
        self.model = inception_v3(pretrained=True, transform_input=False).to(self.device)
        self.model.eval()
        
        # 移除最后的分类层，只保留特征提取部分
        self.model.fc = nn.Identity()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("FID initialized successfully")
    
    def extract_features(self, images: List[Image.Image]) -> np.ndarray:
        """
        提取图像特征
        
        Args:
            images: 图像列表
            
        Returns:
            特征数组
        """
        features = []
        
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
                
                # 提取特征
                batch_features = self.model(batch_tensor)
                features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算特征的均值和协方差矩阵
        
        Args:
            features: 特征数组
            
        Returns:
            均值和协方差矩阵
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_frechet_distance(self, mu1: np.ndarray, sigma1: np.ndarray,
                                 mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """
        计算Frechet距离
        
        Args:
            mu1: 第一个分布的均值
            sigma1: 第一个分布的协方差矩阵
            mu2: 第二个分布的均值
            sigma2: 第二个分布的协方差矩阵
            
        Returns:
            Frechet距离
        """
        # 计算均值差的平方
        diff = mu1 - mu2
        
        # 计算协方差矩阵的乘积的平方根
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # 处理数值不稳定性
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % 1e-6
            logger.warning(msg)
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # 计算Frechet距离
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) + 
                np.trace(sigma2) - 2 * tr_covmean)
    
    def compute_fid(self, generated_images: List[Image.Image], 
                    real_images: List[Image.Image]) -> float:
        """
        计算FID分数
        
        Args:
            generated_images: 生成图像列表
            real_images: 真实图像列表
            
        Returns:
            FID分数
        """
        logger.info(f"Computing FID for {len(generated_images)} generated and {len(real_images)} real images")
        
        # 提取特征
        gen_features = self.extract_features(generated_images)
        real_features = self.extract_features(real_images)
        
        # 计算统计量
        mu_gen, sigma_gen = self.calculate_statistics(gen_features)
        mu_real, sigma_real = self.calculate_statistics(real_features)
        
        # 计算FID
        fid_score = self.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
        
        return float(fid_score)
    
    def compute_score(self, images: List[Image.Image], reference_images: List[Image.Image] = None) -> List[float]:
        """
        计算FID分数（为了与其他指标保持一致的接口）
        
        Args:
            images: 待评估图像列表
            reference_images: 参考图像列表
            
        Returns:
            FID分数列表（每个图像一个分数，这里返回相同的FID值）
        """
        if reference_images is None:
            # 如果没有参考图像，使用自身作为参考（这不是标准做法，仅为接口一致性）
            logger.warning("No reference images provided for FID, using input images as reference")
            reference_images = images
        
        fid_score = self.compute_fid(images, reference_images)
        
        # 返回每个图像相同的FID分数
        return [fid_score] * len(images)
    
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
            'fid_value': scores[0] if scores else 0.0  # FID是单一值
        }
