FROM ubuntu:24.04

WORKDIR /usr/src/app

# python
RUN apt update && apt install gcc g++ curl python3.12 python3.12-venv supervisor -y
COPY requirements.txt .
ENV VIRTUAL_ENV=/usr/src/app/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# julia
RUN curl -fsSL https://install.julialang.org | sh -s -- -y
ENV PATH="/root/.juliaup/bin:$PATH"
ENV julia_version=1.11.5
RUN juliaup add $julia_version && juliaup default $julia_version
COPY requirements.jl .
RUN julia requirements.jl

COPY layer1 layer1
COPY layer2 layer2
COPY layer3 layer3
COPY database database
COPY compute compute

COPY compile.sh compile.jl .
RUN ./compile.sh

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
