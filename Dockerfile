# Multi-Agent RL Cybersecurity System
# Pre-built PyTorch image (GPU/CPU auto-detect)
# Fast build - PyTorch already included!

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="Multi-Agent RL Cybersecurity"
LABEL description="DQN-based threat detection (GPU/CPU auto-detect)"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install ONLY other dependencies (PyTorch already in base image!)
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scikit-learn \
    matplotlib \
    seaborn \
    requests \
    tqdm

# Copy application
COPY . .

# Create output directories
RUN mkdir -p data outputs logs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')" || exit 0

CMD ["python", "main.py"]
