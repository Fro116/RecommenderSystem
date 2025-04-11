FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

WORKDIR /usr/src/app
COPY embed_py embed_py
WORKDIR /usr/src/app/embed_py/notebooks/Finetune
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn", "embed:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"]
