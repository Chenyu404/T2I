"""
数据集加载器
统一管理多个数据集的加载和处理
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image
import json

from .parti_prompts import PartiPromptsDataset
from .t2i_factualbench import T2IFactualBenchDataset
from .t2i_compbench import T2ICompBenchDataset

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    数据集加载器
    统一管理多个数据集
    """
    
    def __init__(self, config: Dict):
        """
        初始化数据集加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.datasets = {}
        
        # 初始化各个数据集
        self._init_datasets()
    
    def _init_datasets(self):
        """初始化数据集"""
        datasets_config = self.config.get('datasets', {})
        
        # 初始化PartiPrompts数据集
        if 'parti_prompts' in datasets_config:
            parti_config = datasets_config['parti_prompts']
            try:
                self.datasets['parti_prompts'] = PartiPromptsDataset(
                    dataset_name=parti_config.get('name', 'nateraw/parti-prompts'),
                    local_path=parti_config.get('local_path', 'data/parti_prompts'),
                    download=parti_config.get('download', True)
                )
                logger.info("PartiPrompts dataset initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PartiPrompts: {e}")
        
        # 初始化T2I-FactualBench数据集
        if 't2i_factualbench' in datasets_config:
            factual_config = datasets_config['t2i_factualbench']
            try:
                self.datasets['t2i_factualbench'] = T2IFactualBenchDataset(
                    dataset_name=factual_config.get('name', 'Sakeoffellow001/T2i_Factualbench'),
                    local_path=factual_config.get('local_path', 'data/t2i_factualbench'),
                    download=factual_config.get('download', True)
                )
                logger.info("T2I-FactualBench dataset initialized")
            except Exception as e:
                logger.error(f"Failed to initialize T2I-FactualBench: {e}")

        # 初始化T2I-CompBench数据集
        if 't2i_compbench' in datasets_config:
            compbench_config = datasets_config['t2i_compbench']
            try:
                self.datasets['t2i_compbench'] = T2ICompBenchDataset(
                    dataset_name=compbench_config.get('name', 'NinaKarine/t2i-compbench'),
                    local_path=compbench_config.get('local_path', 'data/t2i_compbench'),
                    download=compbench_config.get('download', True)
                )
                logger.info("T2I-CompBench dataset initialized")
            except Exception as e:
                logger.error(f"Failed to initialize T2I-CompBench: {e}")
    
    def get_dataset(self, dataset_name: str):
        """
        获取指定数据集
        
        Args:
            dataset_name: 数据集名称
            
        Returns:
            数据集对象
        """
        return self.datasets.get(dataset_name)
    
    def get_all_prompts(self, max_samples_per_dataset: Optional[int] = None) -> List[Tuple[str, str]]:
        """
        获取所有数据集的提示文本
        
        Args:
            max_samples_per_dataset: 每个数据集的最大样本数
            
        Returns:
            (提示文本, 数据集名称)的列表
        """
        all_prompts = []
        
        for dataset_name, dataset in self.datasets.items():
            try:
                prompts = dataset.get_prompts(max_samples_per_dataset)
                for prompt in prompts:
                    all_prompts.append((prompt, dataset_name))
                logger.info(f"Loaded {len(prompts)} prompts from {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to get prompts from {dataset_name}: {e}")
        
        return all_prompts
    
    def get_prompts_by_dataset(self, dataset_name: str, max_samples: Optional[int] = None) -> List[str]:
        """
        获取指定数据集的提示文本
        
        Args:
            dataset_name: 数据集名称
            max_samples: 最大样本数
            
        Returns:
            提示文本列表
        """
        dataset = self.get_dataset(dataset_name)
        if dataset is None:
            logger.error(f"Dataset {dataset_name} not found")
            return []
        
        return dataset.get_prompts(max_samples)
    
    def get_dataset_statistics(self) -> Dict:
        """
        获取所有数据集的统计信息
        
        Returns:
            统计信息字典
        """
        stats = {}
        
        for dataset_name, dataset in self.datasets.items():
            try:
                stats[dataset_name] = dataset.get_statistics()
            except Exception as e:
                logger.error(f"Failed to get statistics for {dataset_name}: {e}")
                stats[dataset_name] = {'error': str(e)}
        
        return stats
    
    def create_evaluation_dataset(self, 
                                dataset_names: List[str] = None,
                                max_samples_per_dataset: int = 100,
                                output_path: str = None) -> List[Dict]:
        """
        创建评估数据集
        
        Args:
            dataset_names: 要包含的数据集名称列表
            max_samples_per_dataset: 每个数据集的最大样本数
            output_path: 输出文件路径
            
        Returns:
            评估数据列表
        """
        if dataset_names is None:
            dataset_names = list(self.datasets.keys())
        
        evaluation_data = []
        
        for dataset_name in dataset_names:
            dataset = self.get_dataset(dataset_name)
            if dataset is None:
                continue
            
            try:
                prompts = dataset.get_prompts(max_samples_per_dataset)
                
                for i, prompt in enumerate(prompts):
                    eval_item = {
                        'id': f"{dataset_name}_{i}",
                        'prompt': prompt,
                        'dataset': dataset_name,
                        'generated_image_path': None,  # 将在生成图像后填充
                        'scores': {}  # 将在评估后填充
                    }
                    evaluation_data.append(eval_item)
                
                logger.info(f"Added {len(prompts)} samples from {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to process {dataset_name}: {e}")
        
        # 保存到文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Evaluation dataset saved to {output_path}")
        
        return evaluation_data
    
    def load_evaluation_dataset(self, input_path: str) -> List[Dict]:
        """
        加载评估数据集
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            评估数据列表
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
        
        logger.info(f"Loaded evaluation dataset with {len(evaluation_data)} samples")
        return evaluation_data
