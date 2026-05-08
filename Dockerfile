# Railway worker Dockerfile
FROM python:3.12-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install pay CLI. The install.sh script may need to be run differently.
# Nix or cargo-based approaches below. Prefer the direct binary if possible.
RUN curl -fsSL https://pay.sh/install.sh -o /tmp/install-pay.sh \
    && sh /tmp/install-pay.sh \
    && found=$(find / -name pay -type f -executable 2>/dev/null | head -1) \
    && if [ -n "$found" ]; then cp "$found" /usr/local/bin/pay && chmod +x /usr/local/bin/pay; fi \
    && (which pay && pay --version) || echo "pay CLI install failed — enrichment disabled"

COPY app/ app/
COPY backend/ backend/
COPY cancerhawk-provider.yml .

EXPOSE 8765

CMD ["python", "-m", "app.main"]
