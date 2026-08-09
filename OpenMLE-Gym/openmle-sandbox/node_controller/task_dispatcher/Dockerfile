FROM docker.m.daocloud.io/library/python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY task_dispatcher.py .

CMD ["python", "-u", "task_dispatcher.py"]
