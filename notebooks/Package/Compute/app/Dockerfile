FROM julia:1.11-alpine
RUN apk add --no-cache \
    build-base \
    g++ \
    curl \
    krb5-libs \
    supervisor
WORKDIR /usr/src/app
COPY requirements.jl .
RUN julia requirements.jl
COPY compute compute
COPY compile.sh compile.jl .
RUN ./compile.sh
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]