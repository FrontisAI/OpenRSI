FROM docker.m.daocloud.io/library/python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api_server.py .

EXPOSE 8000

#CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
