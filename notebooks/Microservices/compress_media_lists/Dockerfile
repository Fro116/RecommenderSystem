FROM python:3.10
WORKDIR /usr/src/app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV TQDM_DISABLE=1
CMD ["gunicorn", "-w", "2", "--config", "notebooks/config.py", "--chdir", "notebooks", "compress_media_lists:app", "-b", "0.0.0.0:8080"]