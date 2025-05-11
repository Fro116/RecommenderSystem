FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04
# TODO use the pytorch images once they catch up to 12.9

RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages
COPY embed_py embed_py
WORKDIR /usr/src/app/embed_py/notebooks/Finetune
CMD ["uvicorn", "embed:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"]
