import os
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
from PIL import Image, ImageEnhance
import torch
import numpy as np
from torchvision import models, transforms
import sys
sys.path.append(os.path.dirname(__file__))
from 垃圾文件.u2net_infer import u2net_infer, load_u2net, download_u2net_model


def get_text_mask(image, device, enhance_contrast=True, contrast_factor=1.5, method='threshold', threshold=80):
    # if method == 'u2net':
    #     model_path = os.path.join(os.path.dirname(__file__), 'u2net.pth')
    #     if not os.path.exists(model_path):
    #         download_u2net_model(model_path)
    #     model = load_u2net(model_path).to(device)
    #     mask_img = u2net_infer(image, model, device)
    if method == 'threshold':
        # 直接灰度阈值分割
        gray = image.convert('L')
        if enhance_contrast:
            gray = ImageEnhance.Contrast(gray).enhance(contrast_factor)
        gray_np = np.array(gray)
        mask_np = (gray_np < threshold).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_np).convert('L').resize(image.size)
    # else:
    #     # DeepLabV3分割
    #     preprocess = transforms.Compose([
    #         transforms.Resize((512, 512)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     input_tensor = preprocess(image).unsqueeze(0).to(device)
    #     with torch.no_grad():
    #         output = models.segmentation.deeplabv3_resnet101(pretrained=True).to(device).eval()(input_tensor)["out"][0]
    #     mask = output.argmax(0).byte().cpu().numpy()
    #     mask = (mask > 0).astype(np.uint8) * 255
    #     mask_img = Image.fromarray(mask).resize(image.size)
    # 灰度图增强
    if method == 'threshold':
        # 二值mask直接转三通道
        enhanced_img = Image.fromarray(np.stack([mask_img]*3, axis=-1))
        return enhanced_img
    gray = image.convert('L')
    if enhance_contrast:
        gray = ImageEnhance.Contrast(gray).enhance(contrast_factor)
    gray_np = np.array(gray)
    mask_np = np.array(mask_img)
    enhanced = np.zeros_like(gray_np)
    enhanced[mask_np > 128] = gray_np[mask_np > 128]
    enhanced[mask_np <= 128] = 255
    enhanced_img = Image.fromarray(enhanced).convert('RGB')
    return enhanced_img


def main(input_dir='input', output_dir='output', prompt='handwriting in oracle bone script style', controlnet_model = "lllyasviel/sd-controlnet-seg", base_model='runwayml/stable-diffusion-v1-5', strength=1, guidance_scale=7.5, threshold=80, structure_dir='structure'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    controlnet = ControlNetModel.from_pretrained(controlnet_model).to(device)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        base_model,
        controlnet=controlnet
    ).to(device)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(structure_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)
            structure_path = os.path.join(structure_dir, fname)
            image = Image.open(input_path).convert('RGB')
            # 直接灰度阈值分割结构图
            control_image = get_text_mask(image, device, method='threshold', threshold=threshold)
            control_image.save(structure_path)
            # ControlNet seg推理
            result = pipe(prompt=prompt, image=image, control_image=control_image, strength=strength, guidance_scale=guidance_scale).images[0]
            bw = result.convert('L')  # 转灰度，但不二值化
            bw.save(output_path)

            print(f"Processed {fname} -> {output_path}, structure saved: {structure_path}")

if __name__ == '__main__':
    main()
