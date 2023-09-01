# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy only download_youtube.py and requirements.txt into the container at /app
COPY download_youtube.py /app/
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx ffmpeg
RUN pip install uvicorn
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8001 available to the world outside this container
EXPOSE 8001

# Command to run the application
CMD ["uvicorn", "download_youtube:app", "--host", "0.0.0.0", "--port", "8001"]
