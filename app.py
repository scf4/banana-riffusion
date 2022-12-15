import base64
import dataclasses
import io
# import logging
from pathlib import Path

import dacite
import PIL
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from riffusion.audio import mp3_bytes_from_wav_bytes, wav_bytes_from_spectrogram_image
from riffusion.datatypes import InferenceInput, InferenceOutput
from riffusion.riffusion_pipeline import RiffusionPipeline

# Global variable for the model pipeline
model = None

# Where built-in seed images are stored
SEED_IMAGES_DIR = Path(Path(__file__).resolve().parent.parent, "seed_images")

repo_id = "riffusion/riffusion-model-v1"


def init():
    global model

    model = RiffusionPipeline.from_pretrained(repo_id, revision="main", torch_dtype=torch.float16, safety_checker=None)

    @dataclasses.dataclass
    class UNet2DConditionOutput:
        sample: torch.FloatTensor

    # Using traced unet from hf hub
    unet_file = hf_hub_download("riffusion/riffusion-model-v1", filename="unet_traced.pt", subfolder="unet_traced")
    unet_traced = torch.jit.load(unet_file)

    class TracedUNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            assert model is not None
            self.in_channels = model.unet.in_channels
            self.device = model.unet.device
            self.dtype = torch.float16

        def forward(self, latent_model_input, t, encoder_hidden_states):
            sample = unet_traced(latent_model_input, t, encoder_hidden_states)[0]
            return UNet2DConditionOutput(sample=sample)

    model.unet = TracedUNet()

    model.to("cuda")


def inference(model_inputs: dict) -> dict:
    global model

    # Parse an InferenceInput dataclass from the payload
    try:
        inputs = dacite.from_dict(InferenceInput, model_inputs)
    except dacite.exceptions.WrongTypeError as exception:
        return {"error": str(exception), "model_inputs": model_inputs}
    except dacite.exceptions.MissingValueError as exception:
        return {"error": str(exception), "model_inputs": model_inputs}

    response = run_model(model, inputs)

    return {"output": response}


def run_model(model, inputs: InferenceInput) -> dict:
    # Load the seed image by ID
    init_image_path = Path(SEED_IMAGES_DIR, f"{inputs.seed_image_id}.png")
    if not init_image_path.is_file():
        return {"error": f"Invalid seed image: {inputs.seed_image_id}"}
    init_image = Image.open(str(init_image_path)).convert("RGB")

    # Load the mask image by ID
    if inputs.mask_image_id:
        mask_image_path = Path(SEED_IMAGES_DIR, f"{inputs.mask_image_id}.png")
        if not mask_image_path.is_file():
            return {"error": f"Invalid mask image: {inputs.mask_image_id}"}
        mask_image = Image.open(str(mask_image_path)).convert("RGB")
    else:
        mask_image = None

    # Execute the model to get the spectrogram image
    image = model.riffuse(inputs, init_image=init_image, mask_image=mask_image)

    # Reconstruct audio from the image
    wav_bytes, duration_s = wav_bytes_from_spectrogram_image(image)
    mp3_bytes = mp3_bytes_from_wav_bytes(wav_bytes)

    # Compute the output as base64 encoded strings
    image_bytes = image_bytes_from_image(image, mode="JPEG")

    # Assemble the output dataclass
    output = InferenceOutput(
        image="data:image/jpeg;base64," + base64_encode(image_bytes),
        audio="data:audio/mpeg;base64," + base64_encode(mp3_bytes),
        duration_s=duration_s,
    )

    return dataclasses.asdict(output)


def image_bytes_from_image(image: Image.Image, mode: str = "PNG") -> io.BytesIO:
    """
    Convert a PIL image into bytes of the given image format.
    """
    image_bytes = io.BytesIO()
    image.save(image_bytes, mode)
    image_bytes.seek(0)
    return image_bytes


def base64_encode(buffer: io.BytesIO) -> str:
    """
    Encode the given buffer as base64.
    """
    return base64.encodebytes(buffer.getvalue()).decode("ascii")
