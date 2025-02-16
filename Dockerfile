FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

ENV AIPROXY_TOKEN=${AIPROXY_TOKEN}

CMD ["python", "main.py"]
