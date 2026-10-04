# CPU image for kind/CI. For GPUs, swap the base for an NVIDIA CUDA runtime
# image and install the matching CUDA torch wheel; the code picks NCCL automatically.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install numpy==2.1.3 prometheus-client==0.21.0 pyyaml==6.0.2
WORKDIR /app
COPY rtrain ./rtrain
RUN useradd -u 10001 trainer && mkdir -p /ckpt && chown trainer /ckpt
USER 10001
ENTRYPOINT ["torchrun"]
