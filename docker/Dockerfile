# Use Python 3.11 slim (smaller image = faster deploy)
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
# libpq-dev = needed for psycopg2 (PostgreSQL client)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching â€” faster rebuilds)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY config.py .
COPY agents/ ./agents/
COPY api/ ./api/
COPY mcp_server/ ./mcp_server/
COPY database/ ./database/

# Create non-root user (security best practice)
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port 8000 (FastAPI)
EXPOSE 8000

# Health check â€” EC2 load balancer pings this
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]