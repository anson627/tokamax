FROM nvidia/cuda:12.9.1-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev python3-venv python3-pip build-essential git curl \
    ca-certificates procps util-linux coreutils bash \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --retries=10 --timeout=120 --no-deps -r /opt/requirements.txt \
    && /opt/venv/bin/python -c "import jax, einshape, numpy; print(jax.__version__)"
ENV PATH="/opt/venv/bin:${PATH}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN useradd --uid 2000 --create-home candidate
RUN mkdir -p /workspace /logs/verifier /opt/moe_eval
COPY . /tests/
RUN cp /tests/worker.py /tests/reference.py /opt/moe_eval/ \
    && chmod -R a+rX,go-w /opt/moe_eval \
    && chmod -R go-rwx /tests \
    && chmod 700 /tests/test.sh
WORKDIR /workspace
