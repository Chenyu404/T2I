"""
环境设置脚本
检查和设置项目运行环境
"""

import os
import sys
import subprocess
import logging
from pathlib import Path


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_python_version():
    """检查Python版本"""
    logger = logging.getLogger(__name__)
    
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    return True


def check_gpu_availability():
    """检查GPU可用性"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {gpu_count} device(s)")
            logger.info(f"Primary GPU: {gpu_name}")
            
            # 检查GPU内存
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU memory: {gpu_memory:.1f} GB")
            
            return True
        else:
            logger.warning("No GPU available, will use CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not installed, cannot check GPU")
        return False


def install_requirements():
    """安装依赖包"""
    logger = logging.getLogger(__name__)
    
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        logger.error(f"Requirements file not found: {requirements_file}")
        return False
    
    try:
        logger.info("Installing requirements...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", requirements_file
        ])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False


def create_directories():
    """创建必要的目录"""
    logger = logging.getLogger(__name__)
    
    directories = [
        "data",
        "data/parti_prompts",
        "data/t2i_factualbench", 
        "data/evalmuse",
        "models",
        "models/cache",
        "results",
        "results/stage1",
        "results/stage2",
        "results/stage3",
        "results/cache",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    return True


def check_huggingface_access():
    """检查Hugging Face访问"""
    logger = logging.getLogger(__name__)
    
    try:
        from transformers import AutoTokenizer
        
        # 尝试加载一个小模型来测试连接
        logger.info("Testing Hugging Face access...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Hugging Face access successful")
        return True
        
    except Exception as e:
        logger.warning(f"Hugging Face access test failed: {e}")
        logger.info("You may need to set up Hugging Face authentication")
        return False


def check_datasets_access():
    """检查datasets库访问"""
    logger = logging.getLogger(__name__)
    
    try:
        from datasets import load_dataset
        
        # 测试加载一个小数据集
        logger.info("Testing datasets library access...")
        # 这里不实际加载，只是测试导入
        logger.info("Datasets library access successful")
        return True
        
    except Exception as e:
        logger.warning(f"Datasets library test failed: {e}")
        return False


def create_config_templates():
    """创建配置文件模板"""
    logger = logging.getLogger(__name__)
    
    # 检查配置文件是否存在
    config_files = [
        "config/stage1_config.yaml",
        "config/stage2_config.yaml", 
        "config/stage3_config.yaml"
    ]
    
    missing_configs = []
    for config_file in config_files:
        if not os.path.exists(config_file):
            missing_configs.append(config_file)
    
    if missing_configs:
        logger.warning(f"Missing config files: {missing_configs}")
        logger.info("Please ensure all configuration files are created")
    else:
        logger.info("All configuration files found")
    
    return len(missing_configs) == 0


def run_system_checks():
    """运行系统检查"""
    logger = logging.getLogger(__name__)
    
    checks = [
        ("Python Version", check_python_version),
        ("GPU Availability", check_gpu_availability),
        ("Directory Creation", create_directories),
        ("Hugging Face Access", check_huggingface_access),
        ("Datasets Access", check_datasets_access),
        ("Config Files", create_config_templates)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        logger.info(f"Running check: {check_name}")
        try:
            result = check_func()
            results[check_name] = result
            status = "PASS" if result else "WARN"
            logger.info(f"{check_name}: {status}")
        except Exception as e:
            results[check_name] = False
            logger.error(f"{check_name}: FAIL - {e}")
    
    return results


def main():
    """主函数"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting environment setup...")
    
    # 运行系统检查
    results = run_system_checks()
    
    # 询问是否安装依赖
    install_deps = input("\nDo you want to install Python dependencies? (y/n): ").lower().strip()
    if install_deps in ['y', 'yes']:
        install_success = install_requirements()
        results["Requirements Installation"] = install_success
    
    # 打印摘要
    print("\n" + "="*50)
    print("ENVIRONMENT SETUP SUMMARY")
    print("="*50)
    
    for check_name, result in results.items():
        status = "✓ PASS" if result else "⚠ WARN/FAIL"
        print(f"{check_name:<25}: {status}")
    
    print("\n" + "="*50)
    
    # 检查关键组件
    critical_checks = ["Python Version", "Directory Creation"]
    critical_passed = all(results.get(check, False) for check in critical_checks)
    
    if critical_passed:
        print("Environment setup completed!")
        print("You can now run the project scripts.")
        
        print("\nNext steps:")
        print("1. Download datasets: python scripts/download_datasets.py")
        print("2. Run Stage 1 evaluation: python scripts/run_stage1.py")
    else:
        print("Critical checks failed. Please fix the issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
