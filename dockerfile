FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

COPY ./app/requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["bash", "./scripts/1_run_rest_api.sh"]