FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
RUN apt-get update && apt-get install -y build-essential
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages
COPY embed_py embed_py
WORKDIR /usr/src/app/embed_py/notebooks/Finetune
CMD ["uvicorn", "embed:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"]
