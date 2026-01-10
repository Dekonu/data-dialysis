# Optimized single-stage build for Data-Dialysis Dashboard API
# Uses BuildKit cache mounts to reduce disk usage during build
# syntax=docker/dockerfile:1.4
FROM python:3.13-slim

WORKDIR /app

# Install dependencies in a single layer with immediate cleanup
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    postgresql-client \
    && pip install --no-cache-dir --upgrade pip \
    && apt-get purge -y gcc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* /var/tmp/*

# Create non-root user early
RUN useradd -m -u 1000 appuser

# Install Python dependencies with cache mount (reduces rebuild time and space)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    pip install --no-cache-dir --user -r requirements.txt \
    && find /root/.local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /root/.local -type f -name "*.pyc" -delete 2>/dev/null || true \
    && find /root/.local -type f -name "*.pyo" -delete 2>/dev/null || true

# Copy application code
COPY src/ ./src/
COPY pyproject.toml pytest.ini ./

# Move dependencies to appuser and clean up in same layer
RUN mv /root/.local /home/appuser/.local \
    && chown -R appuser:appuser /app /home/appuser/.local \
    && find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /app -type f -name "*.pyc" -delete 2>/dev/null || true \
    && find /app -type f -name "*.pyo" -delete 2>/dev/null || true \
    && rm -rf /tmp/* /var/tmp/* /root/.cache 2>/dev/null || true

# Set PATH for appuser
ENV PATH=/home/appuser/.local/bin:$PATH

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/api/health', timeout=5)" || exit 1

# Run application
CMD ["uvicorn", "src.dashboard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

