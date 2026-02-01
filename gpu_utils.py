"""
GPU Configuration and utilities for PyTorch
"""

import torch
import numpy as np


def setup_gpu():
    """Configure PyTorch for GPU usage with RTX cards."""
    
    print("\n" + "="*80)
    print("GPU CONFIGURATION (PyTorch)")
    print("="*80)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        print(f"✓ CUDA available: {cuda_available}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Set default GPU device
        torch.cuda.set_device(0)
        print(f"\n✓ Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Enable faster CUDA kernels
        torch.backends.cudnn.benchmark = True
        print("✓ CUDA benchmarking enabled (faster convolutions)")
        
        # Enable automatic mixed precision (AMP)
        print("✓ Mixed precision training ready (torch.autocast)")
        
        return True
    else:
        print("⚠️  No GPU found. Using CPU instead.")
        print("\nTo use GPU, ensure you have:")
        print("  - NVIDIA GPU (RTX series recommended)")
        print("  - CUDA Toolkit 11.8+ installed")
        print("  - PyTorch with CUDA support: pip install torch torchvision torchaudio")
        return False


def get_device():
    """Get the device (GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def get_device_info():
    """Print detailed device information."""
    
    print("\n" + "="*80)
    print("DEVICE INFORMATION (PyTorch)")
    print("="*80)
    
    print(f"✓ PyTorch Version: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"✓ Number of CPUs: {torch.get_num_threads()}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        
        # Test GPU computation
        device = torch.device('cuda:0')
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        print(f"✓ GPU test computation successful: {c.shape}")
        
        # Clear GPU memory
        del a, b, c
        torch.cuda.empty_cache()
    
    print("="*80)


def empty_cache():
    """Empty GPU cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_gpu_memory_usage():
    """Get GPU memory usage in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        return allocated, reserved
    return 0, 0


def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated, reserved = get_gpu_memory_usage()
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

