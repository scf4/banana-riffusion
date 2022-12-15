# Banana.dev Riffusion template

Run [Riffusion](https://github.com/hmartiro/riffusion-inference) on [https://banana.dev](https://banana.dev)

## Example input

```json
{
  "alpha": 0.75,
  "num_inference_steps": 50,
  "seed_image_id": "og_beat",

  "start": {
    "prompt": "church bells on sunday",
    "seed": 42,
    "denoising": 0.75,
    "guidance": 7.0,
  },

  "end": {
    "prompt": "jazz with piano",
    "seed": 123,
    "denoising": 0.75,
    "guidance": 7.0,
  },
}
```

## Example output

```json
{
  "image": "<base64 encoded JPEG image>",
  "audio": "<base64 encoded MP3 clip>"
}
```
