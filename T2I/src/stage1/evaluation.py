"""
阶段一评估脚本
统一的图文样本评估系统
"""

import os
import sys
import json
import logging
import yaml
from typing import List, Dict, Optional, Tuple
from PIL import Image
import torch
from datetime import datetime
import pandas as pd

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from .metrics import (
    CLIPScore, ImageReward, PickScore, TIFA, FID, InceptionScore
)
from .datasets import DatasetLoader

# 导入工具函数
try:
    from src.utils.device_utils import setup_device
    from src.utils.server_utils import setup_server_matplotlib, apply_server_optimizations
except ImportError:
    # 备用实现
    def setup_device(config):
        device = config.get('device', 'cuda')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config['device'] = device
        return device

    def setup_server_matplotlib():
        pass

    def apply_server_optimizations(config):
        return config

logger = logging.getLogger(__name__)


class Stage1Evaluator:
    """
    阶段一评估器
    统一管理所有评估指标和数据集
    """
    
    def __init__(self, config_path: str = "config/stage1_config.yaml"):
        """
        初始化评估器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()

        # 设置服务器环境
        setup_server_matplotlib()

        # 应用服务器优化
        self.config = apply_server_optimizations(self.config)

        # 初始化设备（自动检测）
        self.device = setup_device(self.config['evaluation'])
        
        # 初始化数据集加载器
        self.dataset_loader = DatasetLoader(self.config)
        
        # 初始化评估指标
        self.metrics = {}
        self._init_metrics()
        
        # 创建输出目录
        self.output_dir = self.config['evaluation']['output_dir']
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("Stage1Evaluator initialized successfully")
    
    def _load_config(self) -> Dict:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def _init_metrics(self):
        """初始化评估指标"""
        metrics_config = self.config['metrics']
        
        # 初始化CLIPScore
        if metrics_config['clip_score']['enabled']:
            try:
                self.metrics['clip_score'] = CLIPScore(
                    model_name=metrics_config['clip_score']['model_name'],
                    device=self.device
                )
                logger.info("CLIPScore initialized")
            except Exception as e:
                logger.error(f"Failed to initialize CLIPScore: {e}")
        
        # 初始化ImageReward
        if metrics_config['image_reward']['enabled']:
            try:
                self.metrics['image_reward'] = ImageReward(
                    model_name=metrics_config['image_reward']['model_name'],
                    device=self.device
                )
                logger.info("ImageReward initialized")
            except Exception as e:
                logger.error(f"Failed to initialize ImageReward: {e}")
        
        # 初始化PickScore
        if metrics_config['pick_score']['enabled']:
            try:
                self.metrics['pick_score'] = PickScore(
                    model_name=metrics_config['pick_score']['model_name'],
                    device=self.device
                )
                logger.info("PickScore initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PickScore: {e}")
        
        # 初始化TIFA
        if metrics_config['tifa']['enabled']:
            try:
                self.metrics['tifa'] = TIFA(
                    model_name=metrics_config['tifa']['model_name'],
                    device=self.device
                )
                logger.info("TIFA initialized")
            except Exception as e:
                logger.error(f"Failed to initialize TIFA: {e}")
        
        # 初始化FID
        if metrics_config['fid']['enabled']:
            try:
                self.metrics['fid'] = FID(
                    batch_size=metrics_config['fid']['batch_size'],
                    dims=metrics_config['fid']['dims'],
                    device=self.device
                )
                logger.info("FID initialized")
            except Exception as e:
                logger.error(f"Failed to initialize FID: {e}")
        
        # 初始化Inception Score
        if metrics_config['inception_score']['enabled']:
            try:
                self.metrics['inception_score'] = InceptionScore(
                    batch_size=metrics_config['inception_score']['batch_size'],
                    splits=metrics_config['inception_score']['splits'],
                    device=self.device
                )
                logger.info("Inception Score initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Inception Score: {e}")
    
    def evaluate_image_text_pairs(self, 
                                 images: List[Image.Image], 
                                 texts: List[str],
                                 reference_images: Optional[List[Image.Image]] = None) -> Dict:
        """
        评估图像-文本对
        
        Args:
            images: 图像列表
            texts: 文本列表
            reference_images: 参考图像列表（用于FID计算）
            
        Returns:
            评估结果字典
        """
        if len(images) != len(texts):
            raise ValueError("Images and texts must have the same length")
        
        logger.info(f"Evaluating {len(images)} image-text pairs")
        
        results = {
            'num_samples': len(images),
            'metrics': {},
            'individual_scores': {metric_name: [] for metric_name in self.metrics.keys()}
        }
        
        batch_size = self.config['evaluation']['batch_size']
        
        # 计算各个指标
        for metric_name, metric in self.metrics.items():
            try:
                logger.info(f"Computing {metric_name}...")
                
                if metric_name in ['clip_score', 'image_reward', 'pick_score', 'tifa']:
                    # 需要图像和文本的指标
                    if hasattr(metric, 'compute_batch_score'):
                        scores = metric.compute_batch_score(images, texts, batch_size)
                    else:
                        scores = metric.compute_score(images, texts)
                
                elif metric_name == 'fid':
                    # FID需要参考图像
                    if reference_images is not None:
                        scores = metric.compute_score(images, reference_images)
                    else:
                        logger.warning("No reference images provided for FID, skipping")
                        continue
                
                elif metric_name == 'inception_score':
                    # IS只需要图像
                    scores = metric.compute_score(images)
                
                else:
                    logger.warning(f"Unknown metric: {metric_name}")
                    continue
                
                # 计算统计信息
                stats = metric.compute_statistics(scores)
                results['metrics'][metric_name] = stats
                results['individual_scores'][metric_name] = scores
                
                logger.info(f"{metric_name} completed: mean={stats['mean']:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to compute {metric_name}: {e}")
                results['metrics'][metric_name] = {'error': str(e)}
        
        return results
    
    def evaluate_datasets(self, 
                         dataset_names: List[str] = None,
                         max_samples_per_dataset: int = 100,
                         generate_images: bool = False) -> Dict:
        """
        评估数据集
        
        Args:
            dataset_names: 要评估的数据集名称列表
            max_samples_per_dataset: 每个数据集的最大样本数
            generate_images: 是否生成图像（如果为False，需要提供现有图像）
            
        Returns:
            评估结果字典
        """
        if dataset_names is None:
            dataset_names = list(self.dataset_loader.datasets.keys())
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'datasets': {}
        }
        
        for dataset_name in dataset_names:
            logger.info(f"Evaluating dataset: {dataset_name}")
            
            try:
                # 获取提示文本
                prompts = self.dataset_loader.get_prompts_by_dataset(
                    dataset_name, max_samples_per_dataset
                )
                
                if not prompts:
                    logger.warning(f"No prompts found for {dataset_name}")
                    continue
                
                # 这里需要生成或加载图像
                # 由于我们没有实际的图像生成模型，创建占位符
                if generate_images:
                    images = self._generate_placeholder_images(prompts)
                else:
                    # 尝试从本地加载图像
                    images = self._load_images_for_prompts(dataset_name, prompts)
                
                if not images:
                    logger.warning(f"No images available for {dataset_name}")
                    continue
                
                # 评估
                dataset_results = self.evaluate_image_text_pairs(images, prompts)
                results['datasets'][dataset_name] = dataset_results
                
                logger.info(f"Completed evaluation for {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset_name}: {e}")
                results['datasets'][dataset_name] = {'error': str(e)}
        
        return results
    
    def _generate_placeholder_images(self, prompts: List[str]) -> List[Image.Image]:
        """
        生成占位符图像（用于测试）
        
        Args:
            prompts: 提示文本列表
            
        Returns:
            占位符图像列表
        """
        logger.info("Generating placeholder images for testing")
        
        images = []
        for i, prompt in enumerate(prompts):
            # 创建简单的彩色占位符图像
            color = (
                (i * 50) % 255,
                (i * 80) % 255,
                (i * 120) % 255
            )
            img = Image.new('RGB', (512, 512), color)
            images.append(img)
        
        return images
    
    def _load_images_for_prompts(self, dataset_name: str, prompts: List[str]) -> List[Image.Image]:
        """
        为提示文本加载对应的图像
        
        Args:
            dataset_name: 数据集名称
            prompts: 提示文本列表
            
        Returns:
            图像列表
        """
        # 这里应该实现从本地目录加载图像的逻辑
        # 目前返回占位符图像
        return self._generate_placeholder_images(prompts)
    
    def save_results(self, results: Dict, filename: str = None):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            filename: 文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # 同时保存为CSV格式（如果有详细分数）
        self._save_results_as_csv(results, output_path.replace('.json', '.csv'))
    
    def _save_results_as_csv(self, results: Dict, csv_path: str):
        """
        将结果保存为CSV格式
        
        Args:
            results: 评估结果
            csv_path: CSV文件路径
        """
        try:
            rows = []
            
            for dataset_name, dataset_results in results.get('datasets', {}).items():
                if 'error' in dataset_results:
                    continue
                
                metrics_data = dataset_results.get('metrics', {})
                for metric_name, metric_stats in metrics_data.items():
                    if 'error' in metric_stats:
                        continue
                    
                    row = {
                        'dataset': dataset_name,
                        'metric': metric_name,
                        'mean': metric_stats.get('mean', 0),
                        'std': metric_stats.get('std', 0),
                        'min': metric_stats.get('min', 0),
                        'max': metric_stats.get('max', 0),
                        'median': metric_stats.get('median', 0),
                        'count': metric_stats.get('count', 0)
                    }
                    rows.append(row)
            
            if rows:
                df = pd.DataFrame(rows)
                df.to_csv(csv_path, index=False)
                logger.info(f"Results also saved as CSV to {csv_path}")
        
        except Exception as e:
            logger.error(f"Failed to save CSV results: {e}")
