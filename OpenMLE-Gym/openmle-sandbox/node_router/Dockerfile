FROM docker.m.daocloud.io/library/python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY gateway.py backends.yaml ./

EXPOSE 6591

CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "6591"]
