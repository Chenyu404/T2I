#!/usr/bin/env python3
"""
服务器路径问题修复脚本
自动检测和修复常见的路径配置问题
"""

import os
import sys
import shutil
import subprocess

def check_project_structure():
    """检查项目结构完整性"""
    print("🔍 检查项目结构...")
    
    required_files = [
        "scripts/run_stage1.py",
        "scripts/run_stage2.py", 
        "scripts/run_stage3.py",
        "src/__init__.py",
        "config/stage1_config.yaml",
        "config/stage2_config.yaml",
        "config/stage3_config.yaml",
        "requirements.txt",
        "run_server.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少以下文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    else:
        print("✅ 项目结构完整")
        return True

def check_python_environment():
    """检查Python环境"""
    print("\n🐍 检查Python环境...")
    
    # 检查Python版本
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python版本过低: {version.major}.{version.minor}")
        print("   需要Python 3.8+")
        return False
    else:
        print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    
    # 检查pip
    try:
        import pip
        print(f"✅ pip可用")
    except ImportError:
        print("❌ pip不可用")
        return False
    
    return True

def check_working_directory():
    """检查工作目录"""
    print("\n📁 检查工作目录...")
    
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"当前工作目录: {current_dir}")
    print(f"脚本所在目录: {script_dir}")
    
    if current_dir != script_dir:
        print("⚠️  工作目录与脚本目录不一致")
        print(f"建议切换到: {script_dir}")
        return False
    else:
        print("✅ 工作目录正确")
        return True

def create_missing_directories():
    """创建缺失的目录"""
    print("\n📂 创建必要目录...")
    
    directories = ['logs', 'results', 'models', 'data', '.cache']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"✅ 创建目录: {directory}")
        else:
            print(f"✅ 目录已存在: {directory}")

def test_script_execution():
    """测试脚本执行"""
    print("\n🧪 测试脚本执行...")
    
    # 测试run_server.py的导入
    try:
        sys.path.insert(0, os.getcwd())
        import run_server
        print("✅ run_server.py 可以正常导入")
        return True
    except Exception as e:
        print(f"❌ run_server.py 导入失败: {e}")
        return False

def generate_fix_commands():
    """生成修复命令"""
    print("\n🔧 生成修复命令...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    commands = [
        f"cd {script_dir}",
        "python test_server_paths.py",
        "python run_server.py --mode install",
        "python run_server.py --mode stage1 --max-samples 10"
    ]
    
    print("请按顺序执行以下命令:")
    for i, cmd in enumerate(commands, 1):
        print(f"{i}. {cmd}")

def main():
    """主函数"""
    print("="*60)
    print("🛠️  服务器路径问题修复工具")
    print("="*60)
    
    all_checks_passed = True
    
    # 执行各项检查
    checks = [
        check_python_environment,
        check_working_directory, 
        check_project_structure,
        test_script_execution
    ]
    
    for check in checks:
        if not check():
            all_checks_passed = False
    
    # 创建必要目录
    create_missing_directories()
    
    print("\n" + "="*60)
    print("📋 修复结果")
    print("="*60)
    
    if all_checks_passed:
        print("🎉 所有检查通过！")
        print("\n可以直接运行:")
        print("python run_server.py --mode all")
    else:
        print("⚠️  发现问题，请按照以下步骤修复:")
        generate_fix_commands()
    
    print("="*60)

if __name__ == "__main__":
    main()
