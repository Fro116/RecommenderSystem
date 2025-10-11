FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /usr/src/app
RUN apt update && apt install gcc g++ curl supervisor build-essential -y

# cloudflare
RUN mkdir -p --mode=0755 /usr/share/keyrings && curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null && echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | tee /etc/apt/sources.list.d/cloudflared.list && apt-get update && apt-get install cloudflared

# python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --break-system-packages

# julia
RUN curl -fsSL https://install.julialang.org | sh -s -- -y
ENV PATH="/root/.juliaup/bin:$PATH"
ENV julia_version=1.11.7
RUN juliaup add $julia_version && juliaup default $julia_version
COPY requirements.jl .
RUN julia requirements.jl

COPY embed_py embed_py
COPY layer1 layer1
COPY layer2 layer2
COPY layer3 layer3
COPY database database
COPY compute compute
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
