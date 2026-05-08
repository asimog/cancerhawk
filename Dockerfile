# Railway worker Dockerfile
FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir --upgrade pip

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY backend/ backend/

EXPOSE 8765

CMD ["python", "-m", "app.main"]
