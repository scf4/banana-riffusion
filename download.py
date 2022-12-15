import dataclasses

import torch
from huggingface_hub import hf_hub_download

from riffusion.riffusion_pipeline import RiffusionPipeline


def download_model():
    repo_id = "riffusion/riffusion-model-v1"
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


if __name__ == "__main__":
    download_model()
