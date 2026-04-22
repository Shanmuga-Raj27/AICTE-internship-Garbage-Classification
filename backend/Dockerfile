# Use Python 3.11 as the base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# NEW UPDATED SECTION: Fixed for Debian Trixie/Hugging Face
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up a non-root user (id 1000) for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Copy the application code
COPY --chown=user . .

# Expose Hugging Face default port
EXPOSE 7860

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]