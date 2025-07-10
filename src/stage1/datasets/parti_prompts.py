"""
PartiPrompts数据集处理
处理来自Hugging Face的PartiPrompts数据集
"""

import os
import json
from typing import List, Dict, Tuple, Optional
from PIL import Image
import requests
from io import BytesIO
import logging

logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not available, using fallback implementation")


class PartiPromptsDataset:
    """
    PartiPrompts数据集处理器
    """
    
    def __init__(self, dataset_name: str = "nateraw/parti-prompts", 
                 local_path: str = "data/parti_prompts", 
                 download: bool = True):
        """
        初始化PartiPrompts数据集
        
        Args:
            dataset_name: Hugging Face数据集名称
            local_path: 本地存储路径
            download: 是否下载数据集
        """
        self.dataset_name = dataset_name
        self.local_path = local_path
        self.download = download
        
        # 创建本地目录
        os.makedirs(local_path, exist_ok=True)
        
        self.dataset = None
        self.data = []
        
        if download:
            self.load_dataset()
    
    def load_dataset(self):
        """加载数据集"""
        try:
            if not DATASETS_AVAILABLE:
                logger.info("datasets library not available, creating fallback data")
                self._create_fallback_data()
                return

            logger.info(f"Loading PartiPrompts dataset: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name)
            
            # 处理数据集
            if 'train' in self.dataset:
                dataset_split = self.dataset['train']
            else:
                # 如果没有train分割，使用第一个可用的分割
                split_name = list(self.dataset.keys())[0]
                dataset_split = self.dataset[split_name]
            
            logger.info(f"Dataset loaded with {len(dataset_split)} samples")
            
            # 转换为内部格式
            for i, item in enumerate(dataset_split):
                # PartiPrompts数据集的实际字段名
                data_item = {
                    'id': i,
                    'prompt': item.get('Prompt', ''),  # 注意大写P
                    'category': item.get('Category', 'unknown'),  # 注意大写C
                    'challenge': item.get('Challenge', 'unknown')  # 注意大写C
                }
                self.data.append(data_item)
            
            logger.info(f"Processed {len(self.data)} prompts")
            
        except Exception as e:
            logger.error(f"Failed to load PartiPrompts dataset: {e}")
            logger.info("Creating fallback data")
            self._create_fallback_data()

    def _create_fallback_data(self):
        """创建备用数据"""
        logger.info("Creating fallback PartiPrompts data")

        fallback_prompts = [
            {
                'id': 0,
                'prompt': 'A red car on the street',
                'category': 'Vehicles',
                'challenge': 'Basic'
            },
            {
                'id': 1,
                'prompt': 'A person reading a book in a library',
                'category': 'People',
                'challenge': 'Basic'
            },
            {
                'id': 2,
                'prompt': 'A beautiful sunset over the ocean',
                'category': 'Nature',
                'challenge': 'Basic'
            },
            {
                'id': 3,
                'prompt': 'A cat sitting on a windowsill',
                'category': 'Animals',
                'challenge': 'Basic'
            },
            {
                'id': 4,
                'prompt': 'A modern building with glass windows',
                'category': 'Architecture',
                'challenge': 'Basic'
            }
        ]

        self.data = fallback_prompts
        logger.info(f"Created {len(self.data)} fallback prompts")
    
    def get_prompts(self, max_samples: Optional[int] = None) -> List[str]:
        """
        获取提示文本列表
        
        Args:
            max_samples: 最大样本数量
            
        Returns:
            提示文本列表
        """
        prompts = [item['prompt'] for item in self.data]
        
        if max_samples is not None:
            prompts = prompts[:max_samples]
        
        return prompts
    
    def get_data_by_category(self, category: str) -> List[Dict]:
        """
        根据类别获取数据
        
        Args:
            category: 类别名称
            
        Returns:
            该类别的数据列表
        """
        return [item for item in self.data if item['category'].lower() == category.lower()]
    
    def get_categories(self) -> List[str]:
        """
        获取所有类别
        
        Returns:
            类别列表
        """
        categories = list(set(item['category'] for item in self.data))
        return sorted(categories)
    
    def get_challenges(self) -> List[str]:
        """
        获取所有挑战类型
        
        Returns:
            挑战类型列表
        """
        challenges = list(set(item['challenge'] for item in self.data))
        return sorted(challenges)
    
    def save_to_json(self, output_path: str):
        """
        保存数据到JSON文件
        
        Args:
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Data saved to {output_path}")
    
    def load_from_json(self, input_path: str):
        """
        从JSON文件加载数据
        
        Args:
            input_path: 输入文件路径
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        logger.info(f"Data loaded from {input_path}")
    
    def get_sample_data(self, num_samples: int = 10) -> List[Dict]:
        """
        获取样本数据
        
        Args:
            num_samples: 样本数量
            
        Returns:
            样本数据列表
        """
        return self.data[:num_samples]
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        categories = {}
        challenges = {}
        
        for item in self.data:
            cat = item['category']
            chal = item['challenge']
            
            categories[cat] = categories.get(cat, 0) + 1
            challenges[chal] = challenges.get(chal, 0) + 1
        
        return {
            'total_samples': len(self.data),
            'categories': categories,
            'challenges': challenges,
            'avg_prompt_length': sum(len(item['prompt']) for item in self.data) / len(self.data) if self.data else 0
        }
