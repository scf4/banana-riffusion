# This file is used to verify your http server acts as expected
# Run it with `python3 test.py``

import base64
from io import BytesIO

import requests
from PIL import Image

model_inputs = {
  "alpha": 0.75,
  "num_inference_steps": 50,
  "seed_image_id": "og_beat",

  "start": {
    "prompt": "lofi hip hop drums with piano",
    "seed": 2496,
    "denoising": 0.75,
    "guidance": 7.0,
  },

  "end": {
    "prompt": "chill vibes electro jazz violin",
    "seed": 69042,
    "denoising": 0.75,
    "guidance": 7.0,
  },
}

res = requests.post("http://localhost:8000/", json=model_inputs)

image_encoded = res.json().output.image.encode("utf-8")
image_bytes = BytesIO(base64.b64decode(image_encoded))
image = Image.open(image_bytes)
image.save("output.jpg")