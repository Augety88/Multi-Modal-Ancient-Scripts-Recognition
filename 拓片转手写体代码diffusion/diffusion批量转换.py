import os
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

prompt = "ancient Chinese oracle bone script, black ink on white paper, brush strokes, realistic handwriting, minimal background, calligraphic style, monochrome"


def main(input_dir='金文拓片', output_dir='金文拓片转手写diffusion', prompt=prompt, strength=0.10, guidance_scale=12, threshold=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5"
    ).to(device)

    for root, dirs, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)
        out_dir = os.path.join(output_dir, rel_dir) if rel_dir != '.' else output_dir
        os.makedirs(out_dir, exist_ok=True)
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                input_path = os.path.join(root, fname)
                output_path = os.path.join(out_dir, fname)
                image = Image.open(input_path).convert('RGB')
                result = pipe(prompt=prompt, image=image, strength=strength, guidance_scale=guidance_scale).images[0]
                # 转为灰度
                bw = result.convert('L')
                # 白底黑字二值化
                bw = bw.point(lambda x: 0 if x > threshold else 255, mode='1')
                bw.save(output_path)
                print(f"Processed {input_path} -> {output_path}")

if __name__ == '__main__':
    main()
