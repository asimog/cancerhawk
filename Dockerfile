# Railway worker Dockerfile
FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pay CLI for x402 payment handling
RUN curl -fsSL https://pay.sh/install.sh | sh \
    && mv /root/.local/bin/pay /usr/local/bin/pay 2>/dev/null || true \
    && mv /root/.cargo/bin/pay /usr/local/bin/pay 2>/dev/null || true \
    && echo "pay CLI available: $(which pay || echo 'not found')"

COPY app/ app/
COPY backend/ backend/
COPY cancerhawk-provider.yml .

EXPOSE 8765 1402

# Start both the pay gateway and the FastAPI worker.
# The gateway on :1402 proxies paid requests to :8765.
# The FastAPI worker on :8765 handles all business logic.
CMD sh -c 'pay --sandbox server start cancerhawk-provider.yml --bind 0.0.0.0:1402 & \
           sleep 2 && \
           python -m app.main'
