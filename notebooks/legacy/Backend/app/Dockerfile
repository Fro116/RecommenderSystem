FROM julia:1.10.2
ENV JULIA_NUM_THREADS=auto
RUN apt-get update -y && apt-get install clang python3 python3-pip python3-venv supervisor -y
WORKDIR /usr/src/app

COPY requirements.txt .
ENV VIRTUAL_ENV=/usr/src/app/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements.jl compile.jl .
RUN julia requirements.jl

COPY import_media_lists import_media_lists
COPY nondirectional nondirectional
COPY bagofwords_jl bagofwords_jl
COPY backend backend
RUN cd import_media_lists && julia ../compile.jl && cd ..
RUN cd nondirectional && julia ../compile.jl && cd ..
RUN cd bagofwords_jl && julia ../compile.jl && cd ..
RUN cd backend && julia ../compile.jl && cd ..

COPY fetch_media_lists fetch_media_lists
COPY bagofwords_py_manga_rating bagofwords_py_manga_rating
COPY bagofwords_py_manga_watch bagofwords_py_manga_watch
COPY bagofwords_py_manga_plantowatch bagofwords_py_manga_plantowatch
COPY bagofwords_py_manga_drop bagofwords_py_manga_drop
COPY bagofwords_py_anime_rating bagofwords_py_anime_rating
COPY bagofwords_py_anime_watch bagofwords_py_anime_watch
COPY bagofwords_py_anime_plantowatch bagofwords_py_anime_plantowatch
COPY bagofwords_py_anime_drop bagofwords_py_anime_drop

COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
