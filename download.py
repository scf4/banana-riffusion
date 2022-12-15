import dataclasses

import torch
from huggingface_hub import hf_hub_download

from riffusion.riffusion_pipeline import RiffusionPipeline


def download_model():
    repo_id = "riffusion/riffusion-model-v1"
    model = RiffusionPipeline.from_pretrained(repo_id, revision="main", torch_dtype=torch.float16, safety_checker=None)

    unet_file = hf_hub_download("riffusion/riffusion-model-v1", filename="unet_traced.pt", subfolder="unet_traced") 
    print(unet_file)

if __name__ == "__main__":
    download_model()
