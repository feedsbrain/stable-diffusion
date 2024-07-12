import os
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
pipe = StableDiffusionPipeline.from_pretrained("Lykon/dreamshaper-8")
pipe = pipe.to(get_torch_device())

if (os.environ.get('USE_GRADIO', "False") == "True"):
    # Serve Web UI
    gr.Interface.from_pipeline(pipe).launch()
else:
    # Define the prompt
    prompt = "A fantasy landscape with mountains and a river, sun behind the clouds"

    # Generate the image
    with torch.no_grad():
        image = pipe(prompt).images[0]

    # Save the generated image
    image.save(f"images/{uuid4()}.png")

    # Display the image (optional)
    image.show()
