"""
T2I-CompBench数据集处理
处理来自Hugging Face的T2I-CompBench数据集
"""

import os
import logging
from typing import List, Dict, Optional
from PIL import Image

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

logger = logging.getLogger(__name__)


class T2ICompBenchDataset:
    """
    T2I-CompBench数据集处理器
    """
    
    def __init__(self, dataset_name: str = "NinaKarine/t2i-compbench",
                 local_path: str = "data/t2i_compbench",
                 download: bool = True):
        """
        初始化T2I-CompBench数据集
        
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

            logger.info(f"Loading T2I-CompBench dataset: {self.dataset_name}")
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
                    'category': item.get('category', 'unknown'),
                    'task_type': item.get('task_type', 'unknown'),
                    'complexity': item.get('complexity', 'unknown')
                }
                
                # 处理图像（如果存在）
                if 'image' in item and item['image'] is not None:
                    data_item['has_reference_image'] = True
                    data_item['reference_image'] = item['image']
                else:
                    data_item['has_reference_image'] = False
                
                self.data.append(data_item)
            
            logger.info(f"Processed {len(self.data)} compositional prompts")
            
        except Exception as e:
            logger.error(f"Failed to load T2I-CompBench dataset: {e}")
            logger.info("Creating fallback data")
            self._create_fallback_data()
    
    def _create_fallback_data(self):
        """创建备用数据"""
        logger.info("Creating fallback T2I-CompBench data")
        
        # 创建一些示例组合任务数据
        fallback_data = [
            {
                'id': 0,
                'prompt': 'A red car and a blue house',
                'category': 'color',
                'task_type': 'attribute_binding',
                'complexity': 'simple',
                'has_reference_image': False
            },
            {
                'id': 1,
                'prompt': 'Three cats sitting on a table',
                'category': 'counting',
                'task_type': 'numeracy',
                'complexity': 'medium',
                'has_reference_image': False
            },
            {
                'id': 2,
                'prompt': 'A dog to the left of a tree',
                'category': 'spatial',
                'task_type': 'spatial_relationship',
                'complexity': 'medium',
                'has_reference_image': False
            },
            {
                'id': 3,
                'prompt': 'A large elephant and a small mouse',
                'category': 'size',
                'task_type': 'attribute_binding',
                'complexity': 'simple',
                'has_reference_image': False
            },
            {
                'id': 4,
                'prompt': 'Two red apples on top of a wooden table next to a green book',
                'category': 'complex',
                'task_type': 'multi_attribute',
                'complexity': 'hard',
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
    
    def get_data_by_task_type(self, task_type: str) -> List[Dict]:
        """
        根据任务类型获取数据
        
        Args:
            task_type: 任务类型
            
        Returns:
            该任务类型的数据列表
        """
        return [item for item in self.data if item['task_type'].lower() == task_type.lower()]
    
    def get_categories(self) -> List[str]:
        """
        获取所有类别
        
        Returns:
            类别列表
        """
        categories = list(set(item['category'] for item in self.data))
        return sorted(categories)
    
    def get_task_types(self) -> List[str]:
        """
        获取所有任务类型
        
        Returns:
            任务类型列表
        """
        task_types = list(set(item['task_type'] for item in self.data))
        return sorted(task_types)
    
    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        
        Returns:
            统计信息字典
        """
        categories = {}
        task_types = {}
        complexities = {}
        
        for item in self.data:
            cat = item['category']
            tt = item['task_type']
            comp = item['complexity']
            
            categories[cat] = categories.get(cat, 0) + 1
            task_types[tt] = task_types.get(tt, 0) + 1
            complexities[comp] = complexities.get(comp, 0) + 1
        
        return {
            'total_samples': len(self.data),
            'categories': categories,
            'task_types': task_types,
            'complexities': complexities,
            'with_reference_images': sum(1 for item in self.data if item['has_reference_image']),
            'avg_prompt_length': sum(len(item['prompt']) for item in self.data) / len(self.data) if self.data else 0
        }
