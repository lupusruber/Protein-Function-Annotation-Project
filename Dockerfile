FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VERSION=12.1

RUN ln -s /usr/bin/python3.10 /usr/bin/python


WORKDIR /ppi

COPY . .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.1+cu121.html && \
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html && \
    pip install torch_geometric

RUN cp requirements.txt requirements_temp.txt

RUN sed -i '/pyg-lib/d' requirements_temp.txt && \
    sed -i '/torch_scatter/d' requirements_temp.txt && \
    sed -i '/torch/d' requirements_temp.txt && \
    sed -i '/torchaudio/d' requirements_temp.txt && \
    sed -i '/torchdata/d' requirements_temp.txt && \
    sed -i '/torchvision/d' requirements_temp.txt && \
    sed -i '/torch_sparse/d' requirements_temp.txt && \
    sed -i '/torch_cluster/d' requirements_temp.txt && \
    sed -i '/torch_spline_conv/d' requirements_temp.txt && \
    sed -i '/dgl/d' requirements_temp.txt

RUN pip install --no-cache-dir -r requirements_temp.txt

CMD ["bash"]
