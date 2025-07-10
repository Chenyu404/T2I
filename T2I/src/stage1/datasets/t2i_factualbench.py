"""
T2I-FactualBench数据集处理
处理来自Hugging Face的T2I-FactualBench数据集
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


class T2IFactualBenchDataset:
    """
    T2I-FactualBench数据集处理器
    """
    
    def __init__(self, dataset_name: str = "Sakeoffellow001/T2i_Factualbench",
                 local_path: str = "data/t2i_factualbench",
                 download: bool = True):
        """
        初始化T2I-FactualBench数据集
        
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

            logger.info(f"Loading T2I-FactualBench dataset: {self.dataset_name}")
            self.dataset = load_dataset(self.dataset_name)
            
            # 处理数据集
            if 'test' in self.dataset:
                dataset_split = self.dataset['test']
            elif 'train' in self.dataset:
                dataset_split = self.dataset['train']
            else:
                # 如果没有标准分割，使用第一个可用的分割
                split_name = list(self.dataset.keys())[0]
                dataset_split = self.dataset[split_name]
            
            logger.info(f"Dataset loaded with {len(dataset_split)} samples")
            
            # 转换为内部格式
            for i, item in enumerate(dataset_split):
                data_item = {
                    'id': i,
                    'prompt': item.get('prompt', item.get('text', '')),
                    'concept': item.get('concept', 'unknown'),
                    'category': item.get('category', 'unknown'),
                    'knowledge_type': item.get('knowledge_type', 'unknown'),
                    'difficulty': item.get('difficulty', 'unknown')
                }
                
                # 处理图像（如果存在）
                if 'image' in item and item['image'] is not None:
                    data_item['has_reference_image'] = True
                    data_item['reference_image'] = item['image']
                else:
                    data_item['has_reference_image'] = False
                
                self.data.append(data_item)
            
            logger.info(f"Processed {len(self.data)} factual prompts")
            
        except Exception as e:
            logger.error(f"Failed to load T2I-FactualBench dataset: {e}")
            # 创建一些示例数据作为备用
            self._create_fallback_data()
    
    def _create_fallback_data(self):
        """创建备用数据"""
        logger.info("Creating fallback T2I-FactualBench data")
        
        fallback_data = [
            {
                'id': 0,
                'prompt': 'The Eiffel Tower in Paris, France',
                'concept': 'Eiffel Tower',
                'category': 'landmark',
                'knowledge_type': 'factual',
                'difficulty': 'easy',
                'has_reference_image': False
            },
            {
                'id': 1,
                'prompt': 'Albert Einstein with his famous equation E=mc²',
                'concept': 'Albert Einstein',
                'category': 'person',
                'knowledge_type': 'historical',
                'difficulty': 'medium',
                'has_reference_image': False
            },
            {
                'id': 2,
                'prompt': 'The Great Wall of China stretching across mountains',
                'concept': 'Great Wall of China',
                'category': 'landmark',
                'knowledge_type': 'factual',
                'difficulty': 'easy',
                'has_reference_image': False
            },
            {
                'id': 3,
                'prompt': 'Leonardo da Vinci painting the Mona Lisa',
                'concept': 'Leonardo da Vinci',
                'category': 'person',
                'knowledge_type': 'historical',
                'difficulty': 'hard',
                'has_reference_image': False
            },
            {
                'id': 4,
                'prompt': 'The Statue of Liberty in New York Harbor',
                'concept': 'Statue of Liberty',
                'category': 'landmark',
                'knowledge_type': 'factual',
                'difficulty': 'easy',
                'has_reference_image': False
            }
        ]
        
        self.data = fallback_data
        logger.info(f"Created {len(self.data)} fallback samples")
    
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
    
    def get_data_by_knowledge_type(self, knowledge_type: str) -> List[Dict]:
        """
        根据知识类型获取数据
        
        Args:
            knowledge_type: 知识类型
            
        Returns:
            该知识类型的数据列表
        """
        return [item for item in self.data if item['knowledge_type'].lower() == knowledge_type.lower()]
    
    def get_categories(self) -> List[str]:
        """
        获取所有类别
        
        Returns:
            类别列表
        """
        categories = list(set(item['category'] for item in self.data))
        return sorted(categories)
    
    def get_knowledge_types(self) -> List[str]:
        """
        获取所有知识类型
        
        Returns:
            知识类型列表
        """
        knowledge_types = list(set(item['knowledge_type'] for item in self.data))
        return sorted(knowledge_types)
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        categories = {}
        knowledge_types = {}
        difficulties = {}
        
        for item in self.data:
            cat = item['category']
            kt = item['knowledge_type']
            diff = item['difficulty']
            
            categories[cat] = categories.get(cat, 0) + 1
            knowledge_types[kt] = knowledge_types.get(kt, 0) + 1
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        return {
            'total_samples': len(self.data),
            'categories': categories,
            'knowledge_types': knowledge_types,
            'difficulties': difficulties,
            'with_reference_images': sum(1 for item in self.data if item['has_reference_image']),
            'avg_prompt_length': sum(len(item['prompt']) for item in self.data) / len(self.data) if self.data else 0
        }
