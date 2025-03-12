FROM ubuntu:24.04
WORKDIR /usr/src/app

# julia
RUN apt update && apt install gcc g++ curl -y
RUN curl -fsSL https://install.julialang.org | sh -s -- -y
ENV PATH="/root/.juliaup/bin:$PATH"
ENV julia_version=1.11.2
RUN juliaup add $julia_version && juliaup default $julia_version
COPY requirements.jl .
RUN julia requirements.jl

COPY api api

COPY compile.sh compile.jl .
RUN ./compile.sh

WORKDIR /usr/src/app/api/notebooks/Finetune
CMD ["julia", "-t", "auto,auto", "-J", "/usr/src/app/sysimg.so", "api.jl", "8080"]
