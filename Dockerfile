FROM wallies/python-cuda:3.10-cuda11.6-runtime

WORKDIR /
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install git
RUN apt-get update && apt-get install -y git build-essential

# Install Python packages
RUN pip3 install --upgrade pip
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
CMD python3 -u server.py
