# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (if needed)
# RUN apt-get update && apt-get install -y <packages>

# CopyCopy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# CopyCopy the rest of the application code
COPY . .

# Set environment variables (if any default values are needed)
# ENV DATA_DIR=/app/data/antibody_antigen_models

# Expose any necessary ports (if applicable)
# EXPOSE 8000

# Define the command to run your application
CMD ["python", "agents.py"]
