FROM python:3.10
WORKDIR /usr/src/app
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV TQDM_DISABLE=1
WORKDIR /usr/src/app/notebooks
CMD ["waitress-serve", "--threads", "1", "--host", "0.0.0.0", "--port", "8080", "transformer:app"]