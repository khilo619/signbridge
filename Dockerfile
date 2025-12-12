FROM python:3.10-slim

WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Add the app root to PYTHONPATH so imports work correctly
# This is CRITICAL for finding your 'CV' and 'api' modules
ENV PYTHONPATH=/app

# Install system dependencies (ffmpeg is needed for video processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-api.txt ./
RUN pip install --no-cache-dir --default-timeout=300 -r requirements-api.txt

# Copy application code
COPY . .

# Expose the default port (Hugging Face Spaces defaults to 7860)
EXPOSE 7860

# Default command
# We use an environment variable so it's easy to change config later
ENV APP_MODULE=api.sign_full.main:app
ENV PORT=7860

# Use shell form to allow variable expansion
CMD ["sh", "-c", "uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT}"]