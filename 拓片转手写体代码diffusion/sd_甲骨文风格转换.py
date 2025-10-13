import os
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

prompt = "ancient Chinese oracle bone script, black ink on white paper, brush strokes, realistic handwriting, minimal background, calligraphic style, monochrome"


def main(input_dir='input', output_dir='output', prompt=prompt, strength=0.10, guidance_scale=12, threshold=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    ).to(device)

    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            input_path = os.path.join(input_dir, fname)
            output_path = os.path.join(output_dir, fname)
            image = Image.open(input_path).convert('RGB')
            result = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale).images[0]
            # 转为灰度
            bw = result.convert('L')
            # 二值化
            bw = bw.point(lambda x: 0 if x > threshold else 255, mode='1')
            bw.save(output_path)
            print(f"Processed {fname} -> {output_path}")

if __name__ == '__main__':
    main()
