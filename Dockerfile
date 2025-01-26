FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /cmmvae

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install essential dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git \
    build-essential cmake python3-dev \
    libgl1 libxext6 wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio torchdata==0.7.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Copy your project code
COPY . .

# Install your Python package dependencies
RUN pip3 install --no-cache-dir .

# Set default runtime command
CMD ["python3", "cmmvae"]