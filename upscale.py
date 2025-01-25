import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# load model and scheduler
model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
pipeline = pipeline.to("cuda")

# let's download an image
# url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
# response = requests.get(url)
# low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
# low_res_img = low_res_img.resize((64, 64)) 

image = Image.open("./rage.png").convert("RGB")
# image = image.resize((64, 64))

prompt = ""

upscaled_image = pipeline(prompt=prompt, image=image).images[0]
upscaled_image.save("upsampled.png")
