"""
服务器环境专用运行脚本
针对无GUI服务器环境优化的项目运行脚本
"""

import os
import sys
import subprocess
import logging
import argparse
from datetime import datetime


def setup_server_environment():
    """设置服务器环境变量"""
    # 设置matplotlib为非交互式后端
    os.environ['MPLBACKEND'] = 'Agg'

    # 禁用wandb交互式登录
    os.environ['WANDB_DISABLED'] = 'true'

    # 设置tokenizers并行处理
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # 设置CUDA相关环境变量
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 设置 Hugging Face 镜像源
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
    
    # 设置缓存目录
    cache_dir = os.path.join(os.getcwd(), '.cache')
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['HF_HOME'] = os.path.join(cache_dir, 'huggingface')
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')
    
    # 设置线程数（避免CPU过载）
    import multiprocessing
    num_cores = multiprocessing.cpu_count()
    thread_count = min(4, num_cores)
    os.environ['OMP_NUM_THREADS'] = str(thread_count)
    os.environ['MKL_NUM_THREADS'] = str(thread_count)


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'logs/server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def run_command(command: list, description: str) -> bool:
    """运行命令"""
    logger = logging.getLogger(__name__)
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(command)}")

    # 确保使用项目根目录作为工作目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Working directory: {project_root}")

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root,  # 使用项目根目录
            env=os.environ.copy()  # 传递环境变量
        )
        
        if result.stdout:
            logger.info(f"Output: {result.stdout}")
        
        logger.info(f"✓ {description} completed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False


def check_server_requirements():
    """检查服务器环境要求"""
    logger = logging.getLogger(__name__)
    logger.info("Checking server requirements...")
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8+ required")
        return False
    
    # 检查磁盘空间
    try:
        import shutil
        total, used, free = shutil.disk_usage('.')
        free_gb = free // (1024**3)
        logger.info(f"Available disk space: {free_gb} GB")
        
        if free_gb < 5:
            logger.warning("Low disk space detected")
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
    
    # 检查GPU
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.device_count()} devices")
        else:
            logger.info("No GPU available, will use CPU")
    except ImportError:
        logger.info("PyTorch not installed yet")
    
    return True


def install_dependencies():
    """安装依赖包"""
    logger = logging.getLogger(__name__)
    logger.info("Installing dependencies for server environment...")
    
    # 基础依赖
    basic_deps = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "transformers>=4.30.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "pandas>=1.5.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "PyYAML>=6.0",
        "requests>=2.28.0"
    ]
    
    for dep in basic_deps:
        command = [sys.executable, "-m", "pip", "install", dep]
        if not run_command(command, f"Installing {dep}"):
            logger.warning(f"Failed to install {dep}, continuing...")
    
    # 可选依赖（失败不影响运行）
    optional_deps = [
        "datasets>=2.12.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.4",
        "psutil>=5.9.0"
    ]
    
    for dep in optional_deps:
        command = [sys.executable, "-m", "pip", "install", dep]
        run_command(command, f"Installing optional {dep}")
    
    return True


def run_project_stage(stage: str, args) -> bool:
    """运行项目阶段"""
    logger = logging.getLogger(__name__)

    # 获取脚本的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if stage == "stage1":
        script_path = os.path.join(script_dir, "scripts", "run_stage1.py")
        command = [
            sys.executable, script_path,
            "--max-samples", str(args.max_samples),
            "--generate-images"
        ]
    elif stage == "stage2":
        script_path = os.path.join(script_dir, "scripts", "run_stage2.py")
        command = [
            sys.executable, script_path, "both"
        ]
    elif stage == "stage3":
        script_path = os.path.join(script_dir, "scripts", "run_stage3.py")
        command = [
            sys.executable, script_path, "demo"  # 只运行演示，避免复杂训练
        ]
    else:
        logger.error(f"Unknown stage: {stage}")
        return False

    # 检查脚本文件是否存在
    if stage in ["stage1", "stage2", "stage3"]:
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Script directory: {script_dir}")
            return False

    return run_command(command, f"Stage {stage}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Run project on server environment')
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['install', 'stage1', 'stage2', 'stage3', 'all'],
                       help='Mode to run')
    parser.add_argument('--max-samples', type=int, default=20,
                       help='Maximum samples for testing (reduced for server)')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # 创建必要目录
    for directory in ['logs', 'results', 'models', 'data', '.cache']:
        os.makedirs(directory, exist_ok=True)
    
    # 设置服务器环境
    setup_server_environment()
    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("="*60)
    logger.info("服务器环境项目运行")
    logger.info("="*60)
    logger.info(f"Arguments: {args}")
    
    # 检查服务器要求
    if not check_server_requirements():
        logger.error("Server requirements check failed")
        sys.exit(1)
    
    success = True
    
    # 安装依赖
    if args.mode in ['install', 'all'] and not args.skip_install:
        success &= install_dependencies()
    
    # 运行阶段
    if args.mode in ['stage1', 'all']:
        success &= run_project_stage('stage1', args)
    
    if args.mode in ['stage2', 'all']:
        success &= run_project_stage('stage2', args)
    
    if args.mode in ['stage3', 'all']:
        success &= run_project_stage('stage3', args)
    
    # 打印结果
    print("\n" + "="*60)
    print("服务器运行结果")
    print("="*60)
    
    if success:
        print("🎉 项目在服务器环境下运行成功！")
        print(f"\n📁 结果文件: {os.path.abspath('results')}")
        print(f"📁 日志文件: {os.path.abspath('logs')}")
        print("\n注意事项:")
        print("- 所有可视化图表已保存为文件，无需GUI显示")
        print("- 模型缓存已设置到本地目录")
        print("- 已针对服务器环境优化批处理大小和线程数")
    else:
        print("❌ 部分功能运行失败")
        print("请检查日志文件获取详细信息")
        sys.exit(1)
    
    print("="*60)


if __name__ == "__main__":
    main()
