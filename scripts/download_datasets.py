"""
数据集下载脚本
下载PartiPrompts和T2I-FactualBench数据集
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.stage1.datasets import PartiPromptsDataset, T2IFactualBenchDataset, T2ICompBenchDataset


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def download_parti_prompts(local_path: str = "data/parti_prompts"):
    """下载PartiPrompts数据集"""
    logger = logging.getLogger(__name__)
    logger.info("Downloading PartiPrompts dataset...")
    
    try:
        dataset = PartiPromptsDataset(
            dataset_name="nateraw/parti-prompts",
            local_path=local_path,
            download=True
        )
        
        # 保存统计信息
        stats = dataset.get_statistics()
        logger.info(f"PartiPrompts statistics: {stats}")
        
        # 保存数据到JSON
        json_path = os.path.join(local_path, "parti_prompts_data.json")
        dataset.save_to_json(json_path)
        
        # 保存样本数据
        sample_data = dataset.get_sample_data(20)
        sample_path = os.path.join(local_path, "sample_data.json")
        
        import json
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        logger.info("PartiPrompts dataset downloaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download PartiPrompts: {e}")
        return False


def download_t2i_factualbench(local_path: str = "data/t2i_factualbench"):
    """下载T2I-FactualBench数据集"""
    logger = logging.getLogger(__name__)
    logger.info("Downloading T2I-FactualBench dataset...")
    
    try:
        dataset = T2IFactualBenchDataset(
            dataset_name="Sakeoffellow001/T2i_Factualbench",
            local_path=local_path,
            download=True
        )
        
        # 保存统计信息
        stats = dataset.get_statistics()
        logger.info(f"T2I-FactualBench statistics: {stats}")
        
        # 保存数据到JSON
        json_path = os.path.join(local_path, "t2i_factualbench_data.json")
        
        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.data, f, ensure_ascii=False, indent=2)
        
        logger.info("T2I-FactualBench dataset downloaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to download T2I-FactualBench: {e}")
        return False


def download_t2i_compbench(local_path: str = "data/t2i_compbench"):
    """下载T2I-CompBench数据集"""
    logger = logging.getLogger(__name__)
    logger.info("Downloading T2I-CompBench dataset...")

    try:
        dataset = T2ICompBenchDataset(
            dataset_name="NinaKarine/t2i-compbench",
            local_path=local_path,
            download=True
        )

        # 保存统计信息
        stats = dataset.get_statistics()
        logger.info(f"T2I-CompBench statistics: {stats}")

        # 保存数据到JSON
        json_path = os.path.join(local_path, "t2i_compbench_data.json")

        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.data, f, ensure_ascii=False, indent=2)

        logger.info("T2I-CompBench dataset downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to download T2I-CompBench: {e}")
        return False


def create_dataset_info():
    """创建数据集信息文件"""
    info = {
        "datasets": {
            "parti_prompts": {
                "name": "PartiPrompts",
                "source": "nateraw/parti-prompts",
                "description": "A collection of prompts for evaluating text-to-image models",
                "url": "https://huggingface.co/datasets/nateraw/parti-prompts",
                "local_path": "data/parti_prompts"
            },
            "t2i_factualbench": {
                "name": "T2I-FactualBench",
                "source": "Sakeoffellow001/T2i_Factualbench",
                "description": "Benchmarking the Factuality of Text-to-Image Models with Knowledge-Intensive Concepts",
                "url": "https://huggingface.co/datasets/Sakeoffellow001/T2i_Factualbench",
                "local_path": "data/t2i_factualbench"
            },
            "t2i_compbench": {
                "name": "T2I-CompBench",
                "source": "NinaKarine/t2i-compbench",
                "description": "A comprehensive benchmark for open-world compositional text-to-image generation",
                "url": "https://huggingface.co/datasets/NinaKarine/t2i-compbench",
                "local_path": "data/t2i_compbench"
            }
        },
        "download_date": datetime.now().isoformat(),
        "version": "1.0.0"
    }
    
    import json
    with open("data/dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Download datasets for Stage 1 evaluation')
    
    parser.add_argument('--datasets', nargs='+',
                       choices=['parti_prompts', 't2i_factualbench', 't2i_compbench', 'all'],
                       default=['all'],
                       help='Datasets to download')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting dataset download")
    logger.info(f"Arguments: {args}")
    
    success_count = 0
    total_count = 0
    
    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['parti_prompts', 't2i_factualbench', 't2i_compbench']
    
    # 下载PartiPrompts
    if 'parti_prompts' in datasets_to_download:
        total_count += 1
        parti_path = os.path.join(args.data_dir, 'parti_prompts')
        if download_parti_prompts(parti_path):
            success_count += 1
    
    # 下载T2I-FactualBench
    if 't2i_factualbench' in datasets_to_download:
        total_count += 1
        factual_path = os.path.join(args.data_dir, 't2i_factualbench')
        if download_t2i_factualbench(factual_path):
            success_count += 1

    # 下载T2I-CompBench
    if 't2i_compbench' in datasets_to_download:
        total_count += 1
        compbench_path = os.path.join(args.data_dir, 't2i_compbench')
        if download_t2i_compbench(compbench_path):
            success_count += 1
    
    # 创建数据集信息文件
    create_dataset_info()
    
    # 打印摘要
    print("\n" + "="*50)
    print("DATASET DOWNLOAD SUMMARY")
    print("="*50)
    print(f"Successfully downloaded: {success_count}/{total_count} datasets")
    
    if success_count == total_count:
        print("All datasets downloaded successfully!")
    else:
        print("Some datasets failed to download. Check logs for details.")
    
    print(f"Data saved to: {os.path.abspath(args.data_dir)}")
    print("="*50)


if __name__ == "__main__":
    main()
