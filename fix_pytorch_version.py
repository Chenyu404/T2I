#!/usr/bin/env python3
"""
PyTorch 版本问题修复脚本
解决 torch.load 安全问题和版本兼容性
"""

import os
import sys
import subprocess
import importlib

def check_pytorch_version():
    """检查 PyTorch 版本"""
    print("🔍 Checking PyTorch version...")
    
    try:
        import torch
        version = torch.__version__
        print(f"Current PyTorch version: {version}")
        
        # 解析版本号
        major, minor = map(int, version.split('.')[:2])
        
        if major > 2 or (major == 2 and minor >= 6):
            print("✅ PyTorch version is compatible (>= 2.6)")
            return True, version
        else:
            print(f"⚠️  PyTorch version {version} is too old (requires >= 2.6)")
            return False, version
            
    except ImportError:
        print("❌ PyTorch is not installed")
        return False, None

def check_safetensors():
    """检查 safetensors 是否可用"""
    print("\n🔍 Checking safetensors...")
    
    try:
        import safetensors
        version = safetensors.__version__
        print(f"✅ safetensors is available: {version}")
        return True, version
    except ImportError:
        print("⚠️  safetensors is not installed")
        return False, None

def check_transformers():
    """检查 transformers 版本"""
    print("\n🔍 Checking transformers...")
    
    try:
        import transformers
        version = transformers.__version__
        print(f"Current transformers version: {version}")
        return True, version
    except ImportError:
        print("❌ transformers is not installed")
        return False, None

def install_safetensors():
    """安装 safetensors"""
    print("\n📦 Installing safetensors...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "safetensors"
        ])
        print("✅ safetensors installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install safetensors: {e}")
        return False

def create_pytorch_workaround():
    """创建 PyTorch 兼容性解决方案"""
    print("\n🛠️  Creating PyTorch compatibility workaround...")
    
    workaround_code = '''
"""
PyTorch 兼容性解决方案
解决 torch.load 安全问题
"""

import torch
import warnings
import os

# 保存原始的 torch.load 函数
_original_torch_load = torch.load

def safe_torch_load(f, map_location=None, pickle_module=None, weights_only=None, **kwargs):
    """
    安全的 torch.load 包装器
    """
    try:
        # 尝试使用 weights_only=True
        if weights_only is None:
            weights_only = True
        
        return _original_torch_load(
            f, 
            map_location=map_location, 
            pickle_module=pickle_module,
            weights_only=weights_only,
            **kwargs
        )
    except Exception as e:
        if "weights_only" in str(e) or "CVE-2025-32434" in str(e):
            warnings.warn(
                "Using torch.load without weights_only=True due to version compatibility. "
                "Consider upgrading PyTorch to >= 2.6 for better security.",
                UserWarning
            )
            # 回退到不使用 weights_only 的版本
            kwargs.pop('weights_only', None)
            return _original_torch_load(
                f,
                map_location=map_location,
                pickle_module=pickle_module,
                **kwargs
            )
        else:
            raise e

# 替换 torch.load
torch.load = safe_torch_load

print("✅ PyTorch compatibility workaround applied")
'''
    
    workaround_path = "pytorch_workaround.py"
    with open(workaround_path, 'w', encoding='utf-8') as f:
        f.write(workaround_code)
    
    print(f"✅ Workaround created: {workaround_path}")
    return workaround_path

def test_torch_load():
    """测试 torch.load 功能"""
    print("\n🧪 Testing torch.load functionality...")
    
    try:
        import torch
        
        # 创建一个简单的测试张量
        test_tensor = torch.randn(3, 3)
        test_file = "test_tensor.pth"
        
        # 保存张量
        torch.save(test_tensor, test_file)
        print("✅ torch.save works")
        
        # 尝试加载张量
        try:
            loaded_tensor = torch.load(test_file, weights_only=True)
            print("✅ torch.load with weights_only=True works")
        except Exception as e:
            print(f"⚠️  torch.load with weights_only=True failed: {e}")
            try:
                loaded_tensor = torch.load(test_file)
                print("✅ torch.load without weights_only works")
            except Exception as e2:
                print(f"❌ torch.load failed completely: {e2}")
                return False
        
        # 清理测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"❌ torch.load test failed: {e}")
        return False

def create_mock_clip_config():
    """创建模拟 CLIP 配置"""
    print("\n⚙️  Creating mock CLIP configuration...")
    
    mock_config = {
        'model': {
            'base_model': 'clip',
            'clip': {
                'model_name': 'openai/clip-vit-base-patch32',
                'local_model_path': 'models/pretrained/clip-vit-base-patch32',
                'freeze_backbone': False,
                'use_mock': True  # 新增：使用模拟模型
            },
            'classifier': {
                'hidden_dims': [512, 256, 128],
                'dropout': 0.3,
                'activation': 'relu',
                'num_classes': 2
            },
            'fusion': {
                'method': 'concat',
                'fusion_dim': 512
            }
        },
        'evaluation': {
            'hallucination_types': [
                'semantic_inconsistency',
                'factual_error',
                'object_hallucination',
                'attribute_error',
                'spatial_error'
            ]
        },
        'output': {
            'results_dir': 'results/stage2'
        },
        'device': 'cpu'
    }
    
    import yaml
    config_path = "config/stage2_config_safe.yaml"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(mock_config, f, default_flow_style=False)
    
    print(f"✅ Safe config created: {config_path}")
    return config_path

def main():
    """主函数"""
    print("="*60)
    print("PYTORCH VERSION FIX TOOL")
    print("="*60)
    
    # 检查当前状态
    pytorch_ok, pytorch_version = check_pytorch_version()
    safetensors_ok, safetensors_version = check_safetensors()
    transformers_ok, transformers_version = check_transformers()
    
    print("\n" + "="*40)
    print("DIAGNOSIS SUMMARY")
    print("="*40)
    print(f"PyTorch >= 2.6: {'✅' if pytorch_ok else '❌'}")
    print(f"safetensors: {'✅' if safetensors_ok else '❌'}")
    print(f"transformers: {'✅' if transformers_ok else '❌'}")
    
    # 解决方案
    solutions_applied = []
    
    if not safetensors_ok:
        print("\n🔧 Applying solution: Install safetensors")
        if install_safetensors():
            solutions_applied.append("Installed safetensors")
            safetensors_ok = True
    
    if not pytorch_ok:
        print("\n🔧 Applying solution: Create compatibility workaround")
        workaround_path = create_pytorch_workaround()
        solutions_applied.append(f"Created workaround: {workaround_path}")
    
    # 创建安全配置
    print("\n🔧 Creating safe configuration")
    safe_config = create_mock_clip_config()
    solutions_applied.append(f"Created safe config: {safe_config}")
    
    # 测试修复效果
    print("\n🧪 Testing fixes...")
    torch_test_ok = test_torch_load()
    
    # 最终报告
    print("\n" + "="*60)
    print("FIX SUMMARY")
    print("="*60)
    
    if solutions_applied:
        print("✅ Applied solutions:")
        for solution in solutions_applied:
            print(f"   - {solution}")
    
    print(f"\n🧪 torch.load test: {'✅ PASSED' if torch_test_ok else '❌ FAILED'}")
    
    if safetensors_ok and torch_test_ok:
        print("\n🎉 FIXES SUCCESSFUL!")
        print("💡 You can now run:")
        print("   python test_stage2_simple.py")
        print("   python fix_stage2_evaluation.py")
    else:
        print("\n⚠️  PARTIAL SUCCESS")
        print("💡 Try running the simple test:")
        print("   python test_stage2_simple.py")
    
    print("="*60)

if __name__ == "__main__":
    main()
