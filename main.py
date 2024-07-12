import torch
import gradio as gr

from uuid import uuid4
from diffusers import StableDiffusionPipeline

def get_torch_device():
  # Check available devices
  if torch.cuda.is_available():
      device = torch.device("cuda")
  elif torch.backends.mps.is_available():
      device = torch.device("mps")
  else:
      device = torch.device("cpu")
  return device

# Load the pre-trained stable diffusion model
model_path = "/Volumes/HOUSEBRAIN/Workspaces/research/stable-diffusion/models/dreamshaper_8.safetensors"

pipe = StableDiffusionPipeline.from_single_file(model_path,torch_dtype=torch.float16,use_safetensors=True)
pipe = pipe.to(get_torch_device())

# gr.Interface.from_pipeline(pipe).launch()

# Define the prompt
prompt = "A fantasy landscape with mountains and a river"

# Generate the image
with torch.no_grad():
    image = pipe(prompt).images[0]

# Save the generated image
image.save(f"images/{uuid4()}.png")

# Display the image (optional)
image.show()
