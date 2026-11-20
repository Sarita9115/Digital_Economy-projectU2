# Yucatán EWRS – Streamlit demo image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies first so this layer is cached across code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the trained model into the image so container startup is instant
RUN python -m models.risk_model

# Run as an unprivileged user
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

# python:slim has no curl, so probe the health endpoint with the stdlib
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
