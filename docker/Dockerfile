FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000 8001 8002 8003 8004 8501

CMD ["python", "main.py", "--help"]