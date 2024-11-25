FROM python:3.9-slim

WORKDIR /app
COPY ./app /app
COPY ./sample.env /app/.env

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

ENTRYPOINT ["bash", "-c", "cd /app && pwd && ls && bash ./scripts/1_run_rest_api.sh"]
