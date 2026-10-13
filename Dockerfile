FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    libgl1 \
    libglib2.0-0 \
    libgl1-mesa-dri \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxcursor1 \
    libxi6 \
    libxinerama1 \
    libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
    --timeout 1000 \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu130 \
    && pip3 install --break-system-packages \
    --timeout 1000 \
    PyYAML \
    matplotlib \
    open3d

WORKDIR /workspace

CMD ["bash"]
