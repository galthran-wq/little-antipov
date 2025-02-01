# Use an official Python runtime as a parent image
FROM python:3.10-slim


WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./requirements/web.txt /app
COPY ./requirements/tg.txt /app
COPY ./requirements/retriever.txt /app

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r web.txt -r tg.txt -r retriever.txt

COPY . /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]