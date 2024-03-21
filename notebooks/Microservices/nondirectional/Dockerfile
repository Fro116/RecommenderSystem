FROM julia:1.10.2
WORKDIR /usr/src/app
COPY . .
RUN apt-get update -y && apt-get install clang -y
RUN julia -t auto requirements.jl
CMD ["julia", "-t", "auto", "-Jsysimg.so", "main.jl", "8080"]