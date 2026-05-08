# Railway worker Dockerfile
FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pay CLI for x402 payment handling.
# The install.sh script downloads a prebuilt binary for Linux amd64/arm64.
RUN curl -fsSL https://pay.sh/install.sh -o /tmp/install-pay.sh \
    && sh /tmp/install-pay.sh \
    && find / -name pay -type f 2>/dev/null | head -5 \
    && (cp /root/.local/bin/pay /usr/local/bin/pay 2>/dev/null \
        || cp /root/.cargo/bin/pay /usr/local/bin/pay 2>/dev/null \
        || cp /usr/local/cargo/bin/pay /usr/local/bin/pay 2>/dev/null \
        || echo "WARNING: pay binary not found after install") \
    && chmod +x /usr/local/bin/pay 2>/dev/null || true \
    && echo "pay CLI available: $(which pay 2>/dev/null || echo 'not found')" \
    && pay --version 2>/dev/null || echo "pay version check failed"

COPY app/ app/
COPY backend/ backend/
COPY cancerhawk-provider.yml .

EXPOSE 8765 1402

CMD sh -c 'python -m app.main'
