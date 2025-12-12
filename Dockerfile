FROM python:3.10-slim

WORKDIR /app

<<<<<<< HEAD
# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Add the app root to PYTHONPATH so imports work correctly
ENV PYTHONPATH=/app

# Install system dependencies
=======
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

>>>>>>> khaled
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

<<<<<<< HEAD
# Install Python dependencies
COPY requirements-api.txt ./
RUN pip install --no-cache-dir --default-timeout=300 -r requirements-api.txt

# Copy application code
COPY . .

# Expose the default port
EXPOSE 7860

# Default command (can be overridden by docker run -e APP_MODULE=...)
# Default to the demo API
ENV APP_MODULE=api.sign_demo.main:app
ENV PORT=7860

# Use shell form to allow variable expansion
CMD ["sh", "-c", "uvicorn ${APP_MODULE} --host 0.0.0.0 --port ${PORT}"]
=======
COPY requirements-api.txt ./

RUN pip install --no-cache-dir --default-timeout=300 -r requirements-api.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
>>>>>>> khaled
