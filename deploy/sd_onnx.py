import os
from diffusers import OnnxStableDiffusionPipeline, OnnxRuntimeModel
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
 
# add the stable diffusion onnx model to model_dir
model_dir = "/onnx_sd/stable-diffusion-v1-4/"
 
prompt = "a photo of an astronaut riding a horse on mars"
 
num_inference_steps = 20
 
scheduler = PNDMScheduler.from_pretrained(os.path.join(model_dir, "scheduler/scheduler_config.json"), use_auth_token='xxx')
 
tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
 
text_encoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "text_encoder/model.onnx")))
 
# in txt to image, vae_encoder is not necessary, only used in image to image generation
# vae_encoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "vae_encoder/model.onnx")))
 
vae_decoder = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "vae_decoder/model.onnx")))
unet = OnnxRuntimeModel(model=OnnxRuntimeModel.load_model(os.path.join(model_dir, "unet/model.onnx")))
 
pipe = OnnxStableDiffusionPipeline(
    vae_encoder=None,
    vae_decoder=vae_decoder,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)
 
image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
 
image.save(f"generated_image.png")