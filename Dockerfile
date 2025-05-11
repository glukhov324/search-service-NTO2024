FROM python:3.10

SHELL [ "/bin/bash", "-c" ]

WORKDIR /app

COPY . .

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    pip install --upgrade pip && \ 
    pip install -r requirements.txt

CMD python3 run.py