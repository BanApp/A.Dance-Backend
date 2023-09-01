# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy only scoring.py and requirements.txt into the container at /app
COPY scoring.py /app/
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg
RUN pip install uvicorn
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8003 available to the world outside this container
EXPOSE 8003

# Command to run the application
CMD ["uvicorn", "scoring:app", "--host", "0.0.0.0", "--port", "8003"]
