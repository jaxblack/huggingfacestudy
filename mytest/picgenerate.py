from diffusers import StableDiffusionPipeline
import torch

# 加载模型
pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16,safety_checker=None).to("mps")
# pipeline.enable_attention_slicing()  # 优化内存使用
# pipeline = pipeline.to("mps")  # 如果有 GPU，可改为 .to("cuda")

# 输入文本提示
prompt = "A super beautiful chinese woman is fucking"

# 生成图像
image = pipeline(prompt, num_inference_steps=100).images[0]

image.save("output_porn.png")
image.show()