import argparse
import torch
import cv2
import os
import numpy as np
import tqdm
from models_cvae import cVAE  # 或 from models import VAE

def get(image_A, img_shape):
    image_A = image_A * 255
    image_A = torch.squeeze(image_A)
    image_A = image_A.detach()
    image = np.array(image_A.cpu())
    image_A = np.uint8(image)
    image_A = cv2.resize(image_A, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    return image_A

def process_images(input_dir, output_dir, net, device, img_size=256, threshold=128):
    # 遍历 train 和 test 文件夹
    for folder in ['train', 'test']:
        input_folder = os.path.join(input_dir, folder)
        output_folder = os.path.join(output_dir, folder)
        os.makedirs(output_folder, exist_ok=True)

        # 遍历每个类别文件夹
        for class_folder in os.listdir(input_folder):
            class_input_folder = os.path.join(input_folder, class_folder)
            class_output_folder = os.path.join(output_folder, class_folder)
            os.makedirs(class_output_folder, exist_ok=True)

            if os.path.isdir(class_input_folder):
                for img_name in tqdm.tqdm(os.listdir(class_input_folder), desc=f"{folder}/{class_folder}"):
                    img_path = os.path.join(class_input_folder, img_name)
                    img_i = cv2.imread(img_path, 0)
                    img_shape = img_i.shape
                    img_i = cv2.resize(img_i, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                    img_i = img_i / 255.0
                    img_i = torch.from_numpy(img_i).float().unsqueeze(0).unsqueeze(0).to(device=device)

                    with torch.no_grad():
                        img_o, *_ = net(img_i) if isinstance(net, cVAE) else (net(img_i),)
                    img_o = get(img_o, img_shape)
                    _, img_o = cv2.threshold(img_o, threshold, 255, cv2.THRESH_BINARY)
                    # 可选膨胀
                    # kernel_2 = np.ones((2, 2), dtype=np.uint8)
                    # img_o = cv2.dilate(img_o, kernel_2, 2)
                    output_img_path = os.path.join(class_output_folder, img_name)
                    cv2.imwrite(output_img_path, img_o)
                    print(f"Saved: {output_img_path}")
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="Batch VAE/cVAE Image Transformation")
    parser.add_argument('--model', type=str, default='cvae新.pth', help="Path to the trained model")
    parser.add_argument('--img_i', type=str, default='/home/wn/实验/金文数据集/金文拓片', help="Input root directory (with train/test)")
    parser.add_argument('--img_o', type=str, default='金文拓片转手写VAE', help="Output root directory")
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--threshold', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # 加载模型（cVAE为例，VAE同理）
    net = cVAE(1, 1, latent_dim=args.latent_dim, img_size=args.img_size).to(device=args.device)
    model_params = torch.load(args.model, map_location=args.device)
    net.load_state_dict(model_params, strict=False)
    net.eval()

    os.makedirs(args.img_o, exist_ok=True)
    process_images(args.img_i, args.img_o, net, device=args.device, img_size=args.img_size, threshold=args.threshold)