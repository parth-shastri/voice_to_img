"""Initilization of the stable diffusion model for text-2-img generation"""
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, PNDMScheduler
import transformers
import torch
import config

print(transformers.__version__)

scheduler = PNDMScheduler.from_pretrained(
    config.DIFFUSION_MODEL_NAME,
    subfolder="scheduler",
    )

# initialize the scheduler
if config.DIFFUSION_SCHEDULER == "EulerScheduler":
    scheduler = EulerDiscreteScheduler.from_pretrained(
        config.DIFFUSION_MODEL_NAME,
        subfolder="scheduler"
        )

# stable pipeline init
pipeline  = StableDiffusionPipeline.from_pretrained(
    config.DIFFUSION_MODEL_NAME,
    scheduler=scheduler,
    revision="fp16",
    torch_dtype = torch.float16
    )

if config.MAKE_LOCAL:
    # scheduler.save_pretrained("models/" + config.DIFFUSION_MODEL_NAME)
    pipeline.save_pretrained("models/" + config.DIFFUSION_MODEL_NAME)


if __name__ == "__main__":
    pipeline.enable_attention_slicing()
    pipeline = pipeline.to("cuda")

    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipeline(prompt, height=512, width=512).images[0]
    image.save("data/astronaut_rides_horse.png")
