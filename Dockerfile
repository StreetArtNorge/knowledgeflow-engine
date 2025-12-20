FROM python:3.11-slim

# Install ffmpeg for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server.py .

# Create downloads directory
RUN mkdir -p downloads

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "server.py"]
