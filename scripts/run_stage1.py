"""
阶段一运行脚本
执行文生图评估系统
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.stage1.evaluation import Stage1Evaluator


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/stage1_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run Stage 1: Text-to-Image Evaluation System')
    
    parser.add_argument('--config', type=str, default='config/stage1_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Dataset names to evaluate (default: all)')
    parser.add_argument('--max-samples', type=int, default=100,
                       help='Maximum samples per dataset')
    parser.add_argument('--generate-images', action='store_true',
                       help='Generate placeholder images for testing')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Stage 1 evaluation")
    logger.info(f"Arguments: {args}")
    
    try:
        # 初始化评估器
        evaluator = Stage1Evaluator(args.config)
        
        # 如果指定了输出目录，更新配置
        if args.output_dir:
            evaluator.output_dir = args.output_dir
            os.makedirs(args.output_dir, exist_ok=True)
        
        # 运行评估
        results = evaluator.evaluate_datasets(
            dataset_names=args.datasets,
            max_samples_per_dataset=args.max_samples,
            generate_images=args.generate_images
        )
        
        # 保存结果
        evaluator.save_results(results)
        
        # 打印摘要
        print("\n" + "="*50)
        print("STAGE 1 EVALUATION SUMMARY")
        print("="*50)
        
        for dataset_name, dataset_results in results.get('datasets', {}).items():
            print(f"\nDataset: {dataset_name}")
            
            if 'error' in dataset_results:
                print(f"  Error: {dataset_results['error']}")
                continue
            
            print(f"  Samples: {dataset_results.get('num_samples', 0)}")
            
            metrics_data = dataset_results.get('metrics', {})
            for metric_name, metric_stats in metrics_data.items():
                if 'error' in metric_stats:
                    print(f"  {metric_name}: Error - {metric_stats['error']}")
                else:
                    mean_score = metric_stats.get('mean', 0)
                    print(f"  {metric_name}: {mean_score:.4f} ± {metric_stats.get('std', 0):.4f}")
        
        print("\n" + "="*50)
        print("Evaluation completed successfully!")
        print(f"Results saved to: {evaluator.output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
