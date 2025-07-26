
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create LanceDB config directory and set environment variable
RUN mkdir -p /app/data/lancedb_config
ENV LANCEDB_CONFIG_DIR=/app/data/lancedb_config

# Set entrypoint
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
