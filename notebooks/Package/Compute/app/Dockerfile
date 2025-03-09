FROM ubuntu:24.04
WORKDIR /usr/src/app

# julia
RUN apt update && apt install gcc g++ curl supervisor -y
RUN curl -fsSL https://install.julialang.org | sh -s -- -y
ENV PATH="/root/.juliaup/bin:$PATH"
ENV julia_version=1.11.2
RUN juliaup add $julia_version && juliaup default $julia_version
COPY requirements.jl .
RUN julia requirements.jl

COPY layer4 layer4
COPY compute compute

COPY compile.sh compile.jl .
RUN ./compile.sh

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
