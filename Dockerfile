# Use an official Python runtime as a parent image
FROM python:3.10-slim


WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt