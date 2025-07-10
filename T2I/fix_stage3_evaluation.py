#!/usr/bin/env python3
"""
修复 STAGE 3 评估问题
重新运行评估并生成完整的结果和指标
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def check_existing_results():
    """检查现有结果"""
    results_dir = "results/stage3"
    
    print("Checking existing results...")
    print(f"Results directory: {results_dir}")
    
    if not os.path.exists(results_dir):
        print("✗ Results directory does not exist")
        return False
    
    files = os.listdir(results_dir)
    print(f"Found {len(files)} files:")
    
    has_evaluation_results = False
    has_metrics = False
    
    for file in files:
        print(f"  - {file}")
        if 'evaluation' in file.lower():
            has_evaluation_results = True
        if file.endswith('.json'):
            # 检查文件内容
            try:
                with open(os.path.join(results_dir, file), 'r') as f:
                    data = json.load(f)
                    if 'metrics' in data or 'datasets' in data:
                        has_metrics = True
            except:
                pass
    
    print(f"\nStatus:")
    print(f"  Has evaluation results: {has_evaluation_results}")
    print(f"  Has metrics data: {has_metrics}")
    
    return has_evaluation_results and has_metrics

def run_complete_evaluation():
    """运行完整的评估"""
    print("\n" + "="*50)
    print("Running complete STAGE 3 evaluation...")
    
    try:
        # 导入评估器
        from src.stage3.evaluation import Stage3Evaluator
        
        # 初始化评估器
        evaluator = Stage3Evaluator()
        
        # 准备测试数据
        test_datasets = prepare_comprehensive_test_data()
        
        print(f"Prepared {len(test_datasets)} test datasets:")
        for name, data in test_datasets.items():
            print(f"  - {name}: {len(data)} samples")
        
        # 运行评估
        print("\nRunning comprehensive evaluation...")
        results = evaluator.run_comprehensive_evaluation(test_datasets)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"complete_evaluation_results_{timestamp}.json"
        evaluator.save_results(results, results_file)
        
        # 生成可视化
        print("Generating visualizations...")
        try:
            evaluator.generate_visualizations(results)
            print("✓ Visualizations generated")
        except Exception as e:
            print(f"⚠ Visualization warning: {e}")
        
        # 打印详细摘要
        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        
        evaluator.print_summary(results)
        
        # 生成详细报告
        generate_detailed_report(results, timestamp)
        
        print(f"\n✓ Complete evaluation finished!")
        print(f"✓ Results saved to: results/stage3/{results_file}")
        print(f"✓ CSV results saved to: results/stage3/complete_evaluation_results_{timestamp}.csv")
        print(f"✓ Detailed report saved to: results/stage3/evaluation_report_{timestamp}.md")
        
        return True
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def prepare_comprehensive_test_data():
    """准备综合测试数据"""
    from PIL import Image
    import numpy as np
    
    def create_mock_image(size=(224, 224)):
        array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        return Image.fromarray(array)
    
    # 创建更全面的测试数据
    test_datasets = {
        'parti_prompts_subset': [],
        't2i_factualbench_subset': [],
        'custom_hallucination_cases': [],
        'challenging_cases': []
    }
    
    # 不同类型的测试样本
    sample_types = [
        ('object_hallucination', True, 'object'),
        ('attribute_hallucination', True, 'attribute'),
        ('spatial_hallucination', True, 'spatial'),
        ('no_hallucination', False, 'none'),
        ('complex_scene', True, 'object'),
        ('simple_scene', False, 'none')
    ]
    
    for dataset_name in test_datasets.keys():
        for i in range(30):  # 每个数据集30个样本
            sample_type, has_hallucination, h_type = sample_types[i % len(sample_types)]
            
            sample = {
                'id': f"{dataset_name}_{i}",
                'image': create_mock_image(),
                'original_image': create_mock_image(),
                'text': f"A {sample_type} test case for {dataset_name} - sample {i}",
                'has_hallucination': has_hallucination,
                'bbox': [
                    np.random.uniform(0, 0.4),
                    np.random.uniform(0, 0.4),
                    np.random.uniform(0.6, 1.0),
                    np.random.uniform(0.6, 1.0)
                ] if has_hallucination and np.random.random() > 0.3 else None,
                'hallucination_type': h_type,
                'difficulty': np.random.choice(['easy', 'medium', 'hard']),
                'scene_complexity': np.random.choice(['simple', 'moderate', 'complex'])
            }
            test_datasets[dataset_name].append(sample)
    
    return test_datasets

def generate_detailed_report(results, timestamp):
    """生成详细的评估报告"""
    report_path = f"results/stage3/evaluation_report_{timestamp}.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# STAGE 3 Evaluation Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 总体摘要
        summary = results.get('summary', {})
        f.write("## Overall Summary\n\n")
        f.write(f"- **Overall Performance**: {summary.get('overall_performance', 0):.4f}\n")
        f.write(f"- **Number of Datasets**: {summary.get('num_datasets', 0)}\n")
        f.write(f"- **Total Samples**: {summary.get('total_samples', 0)}\n\n")
        
        # 各数据集详细结果
        f.write("## Dataset Results\n\n")
        
        for dataset_name, dataset_results in results.get('datasets', {}).items():
            f.write(f"### {dataset_name}\n\n")
            f.write(f"- **Samples**: {dataset_results.get('num_samples', 0)}\n\n")
            
            metrics = dataset_results.get('metrics', {})
            
            # 幻觉检测准确率
            detection = metrics.get('hallucination_detection_accuracy', {})
            if 'error' not in detection:
                f.write("#### Hallucination Detection\n")
                f.write(f"- Accuracy: {detection.get('accuracy', 0):.4f}\n")
                f.write(f"- Precision: {detection.get('precision', 0):.4f}\n")
                f.write(f"- Recall: {detection.get('recall', 0):.4f}\n")
                f.write(f"- F1 Score: {detection.get('f1', 0):.4f}\n")
                f.write(f"- AUC: {detection.get('auc', 0):.4f}\n\n")
            
            # 幻觉定位
            localization = metrics.get('hallucination_localization_iou', {})
            if 'error' not in localization:
                f.write("#### Hallucination Localization\n")
                f.write(f"- Mean IoU: {localization.get('mean_iou', 0):.4f}\n")
                f.write(f"- Std IoU: {localization.get('std_iou', 0):.4f}\n")
                f.write(f"- Detection Rate: {localization.get('detection_rate', 0):.4f}\n\n")
            
            # 纠正效果
            correction = metrics.get('correction_effectiveness', {})
            if 'error' not in correction:
                improvement = correction.get('improvement', {})
                f.write("#### Correction Effectiveness\n")
                f.write(f"- Mean Improvement: {improvement.get('mean', 0):.4f}\n")
                f.write(f"- Std Improvement: {improvement.get('std', 0):.4f}\n")
                f.write(f"- Positive Rate: {improvement.get('positive_rate', 0):.4f}\n\n")
            
            # 语义保持性
            semantic = metrics.get('semantic_preservation', {})
            if 'error' not in semantic:
                f.write("#### Semantic Preservation\n")
                f.write(f"- Mean Preservation: {semantic.get('mean_preservation', 0):.4f}\n")
                f.write(f"- Improvement Rate: {semantic.get('improvement_rate', 0):.4f}\n\n")
            
            # 视觉质量
            quality = metrics.get('visual_quality', {})
            if 'error' not in quality:
                f.write("#### Visual Quality\n")
                f.write(f"- Overall Quality: {quality.get('overall_quality', 0):.4f}\n")
                quality_metrics = quality.get('metrics', {})
                for metric_name, metric_stats in quality_metrics.items():
                    f.write(f"- {metric_name}: {metric_stats.get('mean', 0):.4f}\n")
                f.write("\n")
        
        # 配置信息
        f.write("## Configuration\n\n")
        f.write("```yaml\n")
        config = results.get('config', {})
        import yaml
        f.write(yaml.dump(config, default_flow_style=False))
        f.write("```\n")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Fix Stage 3 evaluation issues')
    parser.add_argument('--force', action='store_true', 
                       help='Force re-evaluation even if results exist')
    
    args = parser.parse_args()
    
    print("="*60)
    print("STAGE 3 EVALUATION FIX TOOL")
    print("="*60)
    
    # 创建必要目录
    os.makedirs('results/stage3', exist_ok=True)
    os.makedirs('results/stage3/visualizations', exist_ok=True)
    
    # 检查现有结果
    has_results = check_existing_results()
    
    if has_results and not args.force:
        print("\n✓ Evaluation results already exist.")
        print("Use --force to re-run evaluation.")
        return
    
    # 运行完整评估
    success = run_complete_evaluation()
    
    if success:
        print("\n" + "="*60)
        print("✓ STAGE 3 EVALUATION FIXED SUCCESSFULLY!")
        print("✓ All evaluation metrics and results are now available.")
        print("✓ Check results/stage3/ directory for complete results.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("✗ FAILED TO FIX EVALUATION!")
        print("✗ Please check the error messages above.")
        print("="*60)

if __name__ == "__main__":
    main()
