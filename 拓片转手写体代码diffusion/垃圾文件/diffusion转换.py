import os
import torch
import numpy as np
from PIL import Image, ImageEnhance
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel

# ---------------- 结构图提取 -------------------
def extract_structure(image, threshold=80, enhance_contrast=True, contrast_factor=1.5):
    # 转灰度
    gray = image.convert('L')
    if enhance_contrast:
        gray = ImageEnhance.Contrast(gray).enhance(contrast_factor)
    gray_np = np.array(gray)
    mask_np = (gray_np < threshold).astype(np.uint8) * 255  # 字变黑，背景白
    mask_img = Image.fromarray(mask_np).convert('RGB')
    return mask_img

# ---------------- 主处理流程 -------------------
def generate_brush_style(
    input_dir='input',
    output_dir='output',
    structure_dir='structure',
    controlnet_model="lllyasviel/sd-controlnet-scribble",
    base_model='runwayml/stable-diffusion-v1-5',
    prompt='handwriting',
    threshold=80,
    guidance_scale=7.5,
    strength=1.0
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载ControlNet
    controlnet = ControlNetModel.from_pretrained(controlnet_model).to(device)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base_model,
        controlnet=controlnet
    ).to(device)
    # pipe.enable_xformers_memory_efficient_attention()  # 可选：节省显存

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(structure_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        input_path = os.path.join(input_dir, fname)
        structure_path = os.path.join(structure_dir, fname)
        output_path = os.path.join(output_dir, fname)

        image = Image.open(input_path).convert('RGB')
        
        # 步骤1：提取结构图
        structure_img = extract_structure(image, threshold=threshold)
        structure_img.save(structure_path)

        # 步骤2：ControlNet生成
        result = pipe(
            prompt=prompt,
            image=image,
            control_image=structure_img,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0]

        # 保存黑白二值图
        bw = result.convert('L').point(lambda x: 255 if x > 1 else 0)
        bw.save(output_path)
        print(f"{fname} processed.")

# ---------------- 程序入口 -------------------
if __name__ == '__main__':
    generate_brush_style()
