# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy only gateway.py and requirements.txt into the container at /app
COPY gateway.py /app/
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install uvicorn
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run gateway.py when the container launches
CMD ["uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8000"]
