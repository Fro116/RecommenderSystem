FROM python:3.10
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["gunicorn", "--preload", "-w", "2", "--threads", "100", "-k", "gevent", "--config", "notebooks/config.py", "--chdir", "notebooks", "fetch_media_lists:app", "-b", "0.0.0.0:8080"]