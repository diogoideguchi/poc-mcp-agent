FROM python:3.14-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN pip install openai
COPY app.py .
# Mantém o container vivo para podermos interagir via terminal
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]