"""
Configuration helper for SD-DeepLab Streamlit App
Detects system capabilities and provides setup recommendations
"""

import os
import sys
import platform
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements"""
    version = sys.version_info
    min_version = (3, 8)
    
    if (version.major, version.minor) >= min_version:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor} - Required: 3.8+")
        return False


def check_cuda_available():
    """Check if CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            print(f"✅ CUDA Available: {device}")
            return True
        else:
            print("⚠️  CUDA not available - using CPU (slower)")
            return False
    except:
        print("❌ PyTorch not installed")
        return False


def check_packages():
    """Check if all required packages are installed"""
    required = [
        'torch', 'torchvision', 'streamlit', 'cv2', 
        'numpy', 'PIL', 'sklearn', 'scipy', 'pandas'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if not missing:
        print("✅ All required packages installed")
        return True
    else:
        print(f"❌ Missing packages: {', '.join(missing)}")
        return False


def check_model_file():
    """Check if model file exists"""
    model_path = Path(__file__).parent / "sddeeplab_final.pth"
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model found: {size_mb:.1f} MB")
        return True
    else:
        print("❌ Model file 'sddeeplab_final.pth' not found")
        return False


def check_architecture_diagram():
    """Check if architecture diagram exists"""
    diagram_path = Path(__file__).parent / "Architecture_diagram.png"
    
    if diagram_path.exists():
        size_kb = diagram_path.stat().st_size / 1024
        print(f"✅ Architecture diagram found: {size_kb:.1f} KB")
        return True
    else:
        print("⚠️  Architecture diagram not found (optional)")
        return False


def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        stat = shutil.disk_usage(Path.home())
        free_gb = stat.free / (1024**3)
        
        if free_gb > 2:
            print(f"✅ Disk space: {free_gb:.1f} GB available")
            return True
        else:
            print(f"⚠️  Low disk space: {free_gb:.1f} GB available (2GB recommended)")
            return False
    except:
        print("⚠️  Could not check disk space")
        return True


def check_memory():
    """Check available RAM"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        if available_gb > 4:
            print(f"✅ RAM available: {available_gb:.1f} GB")
            return True
        else:
            print(f"⚠️  Low RAM: {available_gb:.1f} GB available (4GB recommended)")
            return False
    except:
        print("⚠️  Could not check RAM (psutil not installed)")
        return True


def print_system_info():
    """Print system information"""
    print("\n" + "="*50)
    print("SD-DeepLab System Information")
    print("="*50 + "\n")
    
    # OS
    print(f"📱 Operating System: {platform.system()} {platform.release()}")
    print(f"🖥️  Architecture: {platform.machine()}")
    print(f"📦 Python Executable: {sys.executable}\n")


def run_full_check():
    """Run complete system check"""
    print_system_info()
    
    checks = [
        ("Python Version", check_python_version),
        ("CUDA/GPU", check_cuda_available),
        ("Required Packages", check_packages),
        ("Model File", check_model_file),
        ("Architecture Diagram", check_architecture_diagram),
        ("Disk Space", check_disk_space),
        ("RAM", check_memory)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            result = check_func()
            results[name] = result
        except Exception as e:
            print(f"⚠️  Error checking {name}: {str(e)}")
            results[name] = None
    
    print("\n" + "="*50)
    if all(v for v in results.values() if v is not None):
        print("✅ System ready! Run: streamlit run app.py")
    else:
        print("⚠️  Please install missing components (see above)")
    print("="*50 + "\n")
    
    return results


if __name__ == "__main__":
    run_full_check()
