FROM python:3.14-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY main.py .
CMD ["python", "main.py"]