# LingBot-World Dockerfile with CUDA 12.8 support
FROM nvidia/cuda:12.8.0-cudnn9-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

# Install Hugging Face CLI and authenticate using the command line
RUN huggingface-cli login --token ${HUGGINGFACE_TOKEN} && \
    python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='google/embeddinggemma-300m', filename='config.json', repo_type='model', cache_dir='/app/checkpoints')"

# Copy project files
COPY . .

# Install the package in editable mode
RUN pip install -e .

# Set the default command
CMD ["/bin/bash"]
