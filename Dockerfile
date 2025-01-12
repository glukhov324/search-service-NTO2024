FROM python:3.10

SHELL [ "/bin/bash", "-c" ]

WORKDIR /app

COPY /config /app/config
COPY /src /app/src
COPY ./requirements.txt /app/requirements.txt
COPY ./app.py /app/app.py

RUN apt-get update && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    pip install --upgrade pip && \ 
    pip install -r /app/requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]