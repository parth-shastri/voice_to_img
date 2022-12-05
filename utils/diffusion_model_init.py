"""Initilization of the stable diffusion model for text-2-img generation"""
import os
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, PNDMScheduler
import transformers
import torch
import utils.config as config

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

arg_dict = {
    "revision": "fp16",
    "torch_dtype": torch.float16
}

print(f"Using transformers version {transformers.__version__}")

LOCAL_MODEL_DIR = "models/" + config.DIFFUSION_MODEL_NAME

scheduler = PNDMScheduler.from_pretrained(
    config.DIFFUSION_MODEL_NAME if not os.path.exists(LOCAL_MODEL_DIR) else LOCAL_MODEL_DIR,
    subfolder="scheduler",
    )

# initialize the scheduler
if config.DIFFUSION_SCHEDULER == "EulerScheduler":
    scheduler = EulerDiscreteScheduler.from_pretrained(
        config.DIFFUSION_MODEL_NAME if not os.path.exists(LOCAL_MODEL_DIR) else LOCAL_MODEL_DIR,
        subfolder="scheduler"
        )

# stable pipeline init
pipeline  = StableDiffusionPipeline.from_pretrained(
    config.DIFFUSION_MODEL_NAME if not os.path.exists(LOCAL_MODEL_DIR) else LOCAL_MODEL_DIR,
    scheduler=scheduler,
    **arg_dict if torch.cuda.is_available() else {}
    # revision="fp16",
    # torch_dtype = torch.float16
    )

if config.MAKE_LOCAL and not os.path.exists(LOCAL_MODEL_DIR):
    # scheduler.save_pretrained("models/" + config.DIFFUSION_MODEL_NAME)
    pipeline.save_pretrained(LOCAL_MODEL_DIR)


if __name__ == "__main__":
    pipeline.enable_attention_slicing()
    pipeline = pipeline.to(DEVICE)
    prompt = "a photo of an astronaut riding a horse on mars"
    image = pipeline(prompt, height=512, width=512).images[0]
    image.save("data/astronaut_rides_horse.png")
