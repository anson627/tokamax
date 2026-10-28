FROM nvidia/cuda:12.9.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv python3-pip build-essential git curl \
    ca-certificates procps util-linux coreutils bash ripgrep \
    cuda-nsight-systems-12-9 cuda-nsight-compute-12-9 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --retries=10 --timeout=120 --no-deps -r /opt/requirements.txt \
    && /opt/venv/bin/python -c "import jax, einshape, numpy; print(jax.__version__)"
ENV PATH="/opt/venv/bin:${PATH}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN useradd --uid 2000 --create-home candidate \
    && mkdir -p /workspace/submission /logs/verifier /opt/moe_eval
WORKDIR /workspace
COPY workspace/ /workspace/
RUN mkdir -p /workspace/validation/starter \
    && cp -R /workspace/moe_ffn /workspace/validation/starter/moe_ffn \
    && cp /workspace/validation/worker.py /workspace/validation/reference.py /opt/moe_eval/ \
    && chmod -R a+rX,go-w /workspace/validation /opt/moe_eval \
    && chmod +x /workspace/validation/val.sh
