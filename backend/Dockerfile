# Use Python 3.11 as the base image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Install system dependencies if needed (e.g., for OpenCV or other libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set up a non-root user (id 1000) for Hugging Face Spaces compatibility
RUN useradd -m -u 1000 user
USER user

# Set environment variables for Python
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Copy the application code and models
COPY --chown=user . .

# Expose the default Hugging Face Spaces port
EXPOSE 7860

# Command to run the application
# We use uvicorn to run the FastAPI app on 0.0.0.0:7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
