import os

import torch
from huggingface_hub import hf_hub_download


def download_model():
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    model_id = "stabilityai/stable-diffusion-2-1"

    # Load DPMSolver++ scheduler
    scheduler = DPMSolverMultistepScheduler.from_pretrained(model_id, subfolder="scheduler", use_auth_token=HF_AUTH_TOKEN)

    model = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, scheduler=scheduler, use_auth_token=HF_AUTH_TOKEN)


if __name__ == "__main__":
    download_model()
