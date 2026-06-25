FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 HF_HOME=/app/hf
WORKDIR /app
RUN apt-get update && apt-get install -y git build-essential ninja-build libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" FORCE_CUDA=1 MAX_JOBS=2
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /app/TripoSR
RUN pip install --no-cache-dir setuptools==69.5.1 wheel \
 && pip install --no-cache-dir -r /app/TripoSR/requirements.txt \
 && pip install --no-cache-dir runpod requests onnxruntime pooch
ENV PYTHONPATH=/app/TripoSR
COPY handler.py /app/handler.py
CMD ["python", "-u", "/app/handler.py"]
