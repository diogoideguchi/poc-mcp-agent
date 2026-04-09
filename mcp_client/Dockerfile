FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
# Mantém o container vivo para podermos interagir via terminal
CMD ["python", "main.py"]