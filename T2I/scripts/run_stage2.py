"""
阶段二运行脚本
执行幻觉检测模型训练和评估
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.stage2.train import Stage2Trainer
from src.stage2.evaluate import Stage2Evaluator


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/stage2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def train_model(args):
    """训练模型"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Stage 2 training...")
    
    try:
        trainer = Stage2Trainer(args.config)
        trainer.train()
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


def evaluate_model(args):
    """评估模型"""
    logger = logging.getLogger(__name__)
    logger.info("Starting Stage 2 evaluation...")
    
    try:
        evaluator = Stage2Evaluator(args.config, args.model_path)
        
        # 运行评估
        results = evaluator.evaluate_all_datasets()
        
        # 生成可视化
        if args.generate_plots:
            evaluator.generate_visualizations(results)
        
        # 保存结果
        evaluator.save_results(results)
        
        # 打印摘要
        evaluator.print_summary(results)
        
        logger.info("Evaluation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return False


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run Stage 2: Hallucination Detection Model Training')
    
    parser.add_argument('mode', choices=['train', 'eval', 'both'],
                       help='Mode to run: train, eval, or both')
    parser.add_argument('--config', type=str, default='config/stage2_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Model path for evaluation (default: best model from config)')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs('logs', exist_ok=True)
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Stage 2")
    logger.info(f"Arguments: {args}")
    
    success = True
    
    # 训练模式
    if args.mode in ['train', 'both']:
        success &= train_model(args)
        
        # 如果训练成功且模式是both，设置模型路径为最佳模型
        if success and args.mode == 'both' and args.model_path is None:
            # 从配置中获取模型保存目录
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            model_dir = config['output']['model_save_dir']
            args.model_path = os.path.join(model_dir, 'best_model.pth')
    
    # 评估模式
    if args.mode in ['eval', 'both']:
        # 如果没有指定模型路径，尝试使用默认路径
        if args.model_path is None:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            model_dir = config['output']['model_save_dir']
            default_model_path = os.path.join(model_dir, 'best_model.pth')

            if os.path.exists(default_model_path):
                args.model_path = default_model_path
                logger.info(f"Using default model path: {args.model_path}")
            else:
                logger.warning("No model path specified and no default model found")
                logger.info("Will use mock evaluation mode")

        success &= evaluate_model(args)
    
    # 打印最终结果
    print("\n" + "="*50)
    if success:
        print("STAGE 2 COMPLETED SUCCESSFULLY!")
        
        if args.mode in ['train', 'both']:
            print("✓ Model training completed")
        
        if args.mode in ['eval', 'both']:
            print("✓ Model evaluation completed")
            # 检查是否使用了模拟模式
            if not args.model_path or not os.path.exists(args.model_path):
                print("📋 Note: Used mock evaluation mode (no trained model found)")
                print("💡 To get real results, train the model first: python scripts/run_stage2.py train")

        print(f"Results saved to: results/stage2/")
        
    else:
        print("STAGE 2 FAILED!")
        print("Check logs for details.")
        sys.exit(1)
    
    print("="*50)


if __name__ == "__main__":
    main()
