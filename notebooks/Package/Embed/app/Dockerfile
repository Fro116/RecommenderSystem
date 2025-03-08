FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-cu124.2-4.py310

WORKDIR /usr/src/app
COPY embed_py embed_py
WORKDIR /usr/src/app/embed_py/notebooks/Finetune
CMD ["uvicorn", "embed:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "warning"]
