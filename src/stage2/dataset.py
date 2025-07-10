"""
EvalMuse数据集处理
处理来自Hugging Face的EvalMuse标注数据集

EvalMuse-40K是一个可靠且细粒度的基准数据集，包含全面的人工标注，
用于文本到图像生成模型评估。数据集包含40K个样本，涵盖多种评估维度。
"""

import os
import json
import logging
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor
from datasets import load_dataset
import pandas as pd

logger = logging.getLogger(__name__)


class EvalMuseDataset(Dataset):
    """
    EvalMuse数据集类
    用于幻觉检测模型训练
    """
    
    def __init__(self,
                 dataset_name: str = "DY-Evalab/EvalMuse",
                 local_path: str = "data/evalmuse",
                 split: str = "train",
                 processor=None,
                 max_samples: Optional[int] = None):
        """
        初始化EvalMuse数据集
        
        Args:
            dataset_name: Hugging Face数据集名称
            local_path: 本地存储路径
            split: 数据分割 (train/val/test)
            processor: 图像和文本预处理器
            max_samples: 最大样本数量
        """
        self.dataset_name = dataset_name
        self.local_path = local_path
        self.split = split
        self.processor = processor
        self.max_samples = max_samples
        
        # 创建本地目录
        os.makedirs(local_path, exist_ok=True)
        
        self.data = []
        self.load_dataset()
    
    def load_dataset(self):
        """加载数据集"""
        try:
            logger.info(f"Loading EvalMuse dataset: {self.dataset_name}")
            
            # 尝试加载数据集
            try:
                dataset = load_dataset(self.dataset_name)
                
                # 选择合适的分割
                if self.split in dataset:
                    dataset_split = dataset[self.split]
                elif 'train' in dataset:
                    dataset_split = dataset['train']
                else:
                    dataset_split = dataset[list(dataset.keys())[0]]
                
                logger.info(f"Dataset loaded with {len(dataset_split)} samples")
                
                # 处理数据
                for i, item in enumerate(dataset_split):
                    if self.max_samples and i >= self.max_samples:
                        break
                    
                    # 提取数据项
                    data_item = self._process_item(item, i)
                    if data_item:
                        self.data.append(data_item)
                
            except Exception as e:
                logger.warning(f"Failed to load from HuggingFace: {e}")
                logger.info("Creating fallback EvalMuse data")
                self._create_fallback_data()
            
            logger.info(f"Processed {len(self.data)} samples for {self.split}")
            
        except Exception as e:
            logger.error(f"Failed to load EvalMuse dataset: {e}")
            raise
    
    def _process_item(self, item: Dict, index: int) -> Optional[Dict]:
        """
        处理单个数据项
        
        Args:
            item: 原始数据项
            index: 索引
            
        Returns:
            处理后的数据项
        """
        try:
            # 提取文本提示
            prompt = item.get('prompt', item.get('text', ''))
            if not prompt:
                return None
            
            # 提取图像
            image = item.get('image')
            if image is None:
                return None
            
            # 提取标签（是否有幻觉）
            # 根据EvalMuse数据集的实际结构调整
            label = item.get('has_hallucination', item.get('label', 0))
            if isinstance(label, str):
                label = 1 if label.lower() in ['true', 'yes', '1', 'hallucination'] else 0
            
            # 提取幻觉类型（如果有）
            hallucination_type = item.get('hallucination_type', 'unknown')
            
            # 提取评分（如果有）
            score = item.get('score', item.get('rating', 0.0))
            
            return {
                'id': f"{self.split}_{index}",
                'prompt': prompt,
                'image': image,
                'label': int(label),
                'hallucination_type': hallucination_type,
                'score': float(score)
            }
            
        except Exception as e:
            logger.warning(f"Failed to process item {index}: {e}")
            return None
    
    def _create_fallback_data(self):
        """创建备用数据"""
        logger.info("Creating fallback EvalMuse data")
        
        # 创建一些示例数据
        fallback_samples = [
            {
                'prompt': 'A red car on the street',
                'label': 0,  # 无幻觉
                'hallucination_type': 'none',
                'score': 0.9
            },
            {
                'prompt': 'A blue elephant flying in the sky',
                'label': 1,  # 有幻觉（大象不会飞）
                'hallucination_type': 'factual_error',
                'score': 0.2
            },
            {
                'prompt': 'The Eiffel Tower in London',
                'label': 1,  # 有幻觉（埃菲尔铁塔在巴黎）
                'hallucination_type': 'factual_error',
                'score': 0.1
            },
            {
                'prompt': 'A person reading a book',
                'label': 0,  # 无幻觉
                'hallucination_type': 'none',
                'score': 0.8
            },
            {
                'prompt': 'A cat with three heads',
                'label': 1,  # 有幻觉（猫通常只有一个头）
                'hallucination_type': 'object_hallucination',
                'score': 0.3
            }
        ]
        
        # 根据split创建不同数量的数据
        num_samples = {
            'train': 100,
            'val': 20,
            'test': 20
        }.get(self.split, 50)
        
        for i in range(num_samples):
            base_sample = fallback_samples[i % len(fallback_samples)]
            
            # 创建简单的占位符图像
            color = (
                (i * 50) % 255,
                (i * 80) % 255,
                (i * 120) % 255
            )
            image = Image.new('RGB', (224, 224), color)
            
            data_item = {
                'id': f"{self.split}_{i}",
                'prompt': f"{base_sample['prompt']} (sample {i})",
                'image': image,
                'label': base_sample['label'],
                'hallucination_type': base_sample['hallucination_type'],
                'score': base_sample['score']
            }
            
            self.data.append(data_item)
        
        logger.info(f"Created {len(self.data)} fallback samples")
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取单个样本"""
        item = self.data[idx]

        # 预处理图像和文本
        if self.processor:
            try:
                # 确保图像是PIL Image格式
                image = item['image']
                if not isinstance(image, Image.Image):
                    # 如果是其他格式，尝试转换
                    if hasattr(image, 'convert'):
                        image = image.convert('RGB')
                    else:
                        # 创建占位符图像
                        image = Image.new('RGB', (224, 224), (128, 128, 128))

                inputs = self.processor(
                    text=item['prompt'],
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                )

                return {
                    'input_ids': inputs['input_ids'].squeeze(0),
                    'attention_mask': inputs['attention_mask'].squeeze(0),
                    'pixel_values': inputs['pixel_values'].squeeze(0),
                    'label': torch.tensor(item['label'], dtype=torch.long),
                    'hallucination_type': item['hallucination_type'],
                    'score': torch.tensor(item['score'], dtype=torch.float),
                    'id': item['id']
                }
            except Exception as e:
                logger.warning(f"Failed to process item {idx}: {e}")
                # 返回默认值
                return {
                    'input_ids': torch.zeros(77, dtype=torch.long),
                    'attention_mask': torch.zeros(77, dtype=torch.long),
                    'pixel_values': torch.zeros(3, 224, 224, dtype=torch.float),
                    'label': torch.tensor(0, dtype=torch.long),
                    'hallucination_type': 'unknown',
                    'score': torch.tensor(0.0, dtype=torch.float),
                    'id': item['id']
                }
        else:
            return item
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if not self.data:
            return {}
        
        labels = [item['label'] for item in self.data]
        hallucination_types = [item['hallucination_type'] for item in self.data]
        scores = [item['score'] for item in self.data]
        
        # 统计标签分布
        label_counts = {0: 0, 1: 0}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # 统计幻觉类型分布
        type_counts = {}
        for h_type in hallucination_types:
            type_counts[h_type] = type_counts.get(h_type, 0) + 1
        
        return {
            'total_samples': len(self.data),
            'label_distribution': label_counts,
            'hallucination_types': type_counts,
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'score_range': [min(scores), max(scores)] if scores else [0, 0]
        }


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        训练、验证、测试数据加载器
    """
    # 初始化处理器
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset_config = config['dataset']['evalmuse']
    
    # 创建数据集
    train_dataset = EvalMuseDataset(
        dataset_name=dataset_config['name'],
        local_path=dataset_config['local_path'],
        split='train',
        processor=processor
    )
    
    val_dataset = EvalMuseDataset(
        dataset_name=dataset_config['name'],
        local_path=dataset_config['local_path'],
        split='val',
        processor=processor
    )
    
    test_dataset = EvalMuseDataset(
        dataset_name=dataset_config['name'],
        local_path=dataset_config['local_path'],
        split='test',
        processor=processor
    )
    
    # 创建数据加载器
    batch_size = config['training']['batch_size']
    num_workers = config.get('num_workers', 4)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.get('pin_memory', True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.get('pin_memory', True)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.get('pin_memory', True)
    )
    
    return train_loader, val_loader, test_loader
