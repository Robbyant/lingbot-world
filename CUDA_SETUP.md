# CUDA 12.8 and PyTorch Setup Guide

This guide covers setting up LingBot-World with CUDA 12.8 and PyTorch.

## Prerequisites

- NVIDIA GPU with compute capability 7.0+ (e.g., RTX 2000 series or newer)
- Docker with NVIDIA Container Toolkit installed
- At least 16GB of system RAM
- Sufficient disk space for models and data

## Option 1: Docker Setup (Recommended)

### 1. Install Docker and NVIDIA Container Toolkit

**Ubuntu/Debian:**
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Windows:**
- Install Docker Desktop for Windows
- Ensure WSL2 backend is enabled
- Install NVIDIA drivers for WSL2

### 2. Build and Run with Docker Compose

```bash
# Build the image
docker-compose build

# Run the container
docker-compose up -d

# Enter the container
docker-compose exec lingbot-world bash

# Or run directly
docker-compose run --rm lingbot-world python your_script.py
```

### 3. Verify CUDA Installation

Inside the container:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

## Option 2: Local Installation

### 1. Install CUDA 12.8

**Ubuntu/Debian:**
```bash
# Download and install CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-8

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

**Windows:**
1. Download CUDA 12.8 from [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
2. Run the installer and follow the prompts
3. Verify installation: `nvcc --version`

### 2. Create Python Environment

```bash
# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch with CUDA 12.4 Support

**Note:** PyTorch doesn't have official CUDA 12.8 builds yet. Using CUDA 12.4 builds which are compatible:

```bash
# Install PyTorch with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install LingBot-World Dependencies

```bash
# Install remaining dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

### 5. Verify Installation

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Troubleshooting

### CUDA Version Mismatch

If you encounter CUDA version mismatch errors:
- PyTorch CUDA 12.4 builds are compatible with CUDA 12.x runtime
- Ensure your NVIDIA driver supports CUDA 12.8 (driver version ≥ 550.x)

### Out of Memory Errors

If you encounter OOM errors:
- Reduce batch size in your training scripts
- Enable gradient checkpointing
- Use mixed precision training (fp16 or bf16)

### Flash Attention Installation Issues

If `flash_attn` fails to install:
```bash
# Install with CUDA support
pip install flash-attn --no-build-isolation
```

Or build from source:
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
python setup.py install
```

## Performance Optimization

### Enable TF32 for Ampere+ GPUs

```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Use Compilation (PyTorch 2.x)

```python
model = torch.compile(model)
```

## Additional Resources

- [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
