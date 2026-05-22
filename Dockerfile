# Use Python 3.11 as the base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Install system dependencies needed for libraries (OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the backend requirements file and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up a non-root user (id 1000) for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user

# Set environment variables for the user and Python execution
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Copy the backend application files under user ownership
COPY --chown=user backend/ .

# Expose the default port expected by Hugging Face Spaces
EXPOSE 7860

# Start the FastAPI application with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
