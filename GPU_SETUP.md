# GPU Setup Guide

## Prerequisites

To run this project with GPU acceleration, you need:

### Hardware
- **NVIDIA GPU** with CUDA Compute Capability 3.5 or higher
- **Recommended**: RTX 3060 / RTX 4060 or newer for best performance

### Software Stack

#### 1. NVIDIA CUDA Toolkit (11.8+)
Download and install from: https://developer.nvidia.com/cuda-downloads

```bash
# Verify CUDA installation
nvcc --version
```

#### 2. NVIDIA cuDNN (8.6+)
Download from: https://developer.nvidia.com/cudnn

Extract and add to system PATH

#### 3. Python Packages

Two installation options:

**Option A: Full GPU Support (Recommended)**
```bash
pip install tensorflow[and-cuda]>=2.13.0
```

**Option B: CPU Only**
```bash
pip install tensorflow>=2.13.0
```

## Quick Installation

### Windows
```bash
# Clone or download the project
cd multi_agent_rl_cybersecurity

# Install with GPU support
pip install -r requirements.txt

# Run with GPU
python main.py
```

### Linux/macOS
```bash
# Install with GPU support
pip install tensorflow[and-cuda]>=2.13.0 numpy pandas scikit-learn requests matplotlib seaborn

# Run with GPU
python main.py
```

## Verification

The script will automatically:
1. Detect available GPUs
2. Enable GPU memory growth (prevents OOM)
3. Configure mixed precision training
4. Print GPU device information
5. Verify GPU is working

**Expected Output:**
```
================================================================================
GPU CONFIGURATION
================================================================================
✓ Found 1 GPU(s)
  - PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

✓ GPU memory growth enabled
✓ Using GPU: PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')

================================================================================
DEVICE INFORMATION
================================================================================
✓ TensorFlow: 2.13.0
✓ CUDA available: True
✓ GPU support: True
✓ CPUs: 8
✓ GPUs: 1
✓ GPU test computation successful: (2, 2)
================================================================================
```

## Troubleshooting

### "No GPU found" Warning
This means either:
1. CUDA/cuDNN not installed
2. Wrong TensorFlow version installed
3. GPU not supported

**Solution:**
```bash
# Reinstall TensorFlow for GPU
pip uninstall tensorflow
pip install tensorflow[and-cuda]

# Verify CUDA/cuDNN paths are set
echo %CUDA_PATH%  # Windows
echo $CUDA_PATH   # Linux
```

### CUDA Out of Memory (OOM)
The script automatically enables memory growth, but if issues persist:

Edit `config.py`:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### TensorFlow doesn't detect GPU
```bash
# Test GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Should output:
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## Performance Tips

### Mixed Precision Training (Default)
Enabled by default for ~2x faster training:
```python
# In config.py
MIXED_PRECISION = True
```

### Batch Size Optimization
For GPU, use larger batch sizes:
```python
# In config.py
BATCH_SIZE = 64  # For GPU (default 32)
BATCH_SIZE = 16  # For older GPUs or lower VRAM
```

### Multi-GPU Support
For systems with multiple GPUs:
```python
# TensorFlow handles this automatically with MirroredStrategy
# No code changes needed!
```

## Performance Comparison

**Expected speedup with GPU:**
- Training time: 5-10 minutes (GPU) vs 30-60 minutes (CPU)
- Throughput: 1000+ samples/sec (GPU) vs 100-200 samples/sec (CPU)

## System Requirements

### Minimum
- GPU VRAM: 2GB
- RAM: 8GB
- Storage: 2GB

### Recommended
- GPU VRAM: 4GB+
- RAM: 16GB
- Storage: 5GB
- GPU: RTX 3060 or newer

## Additional Resources

- [TensorFlow GPU Guide](https://www.tensorflow.org/install/gpu)
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-windows/)
- [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/)

## Support

For GPU-related issues:
1. Check TensorFlow GPU documentation
2. Verify CUDA/cuDNN installation
3. Test with `tf.test.is_gpu_available()`
4. Check NVIDIA GPU temperature with `nvidia-smi`

```bash
# Monitor GPU usage during training
nvidia-smi -l 1  # Updates every 1 second
```
