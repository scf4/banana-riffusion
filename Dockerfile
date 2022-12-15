FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install Python packages
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Server boilerplate
ADD server.py .
EXPOSE 8000

# Add riffusion dir
ADD riffusion ./riffusion

# Add seed_images dir
ADD seed_images ./seed_images

# Download model
ADD download.py .
RUN python3 download.py

# App.py with init() and inference()
ADD app.py .

# Start server
# RUN echo "Successfully built Docker image"
CMD python3 -u server.py
