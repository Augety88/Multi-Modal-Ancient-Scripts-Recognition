import argparse
import torch
import cv2
import os
import numpy as np
import tqdm
from 垃圾文件.models_adafunet import UNet1,Generator
from 垃圾文件.models import *

def get(image_A, img_shape):
    image_A = image_A * 255
    image_A = torch.squeeze(image_A)
    image_A = image_A.detach()
    image = np.array(image_A.cpu())
    image_A = np.uint8(image)
    image_A = cv2.resize(image_A, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    return image_A

def process_images(input_dir, output_dir, net_BTOA, device):
    # 遍历 train 和 val 文件夹
    for folder in ['train', 'test']:
        input_folder = os.path.join(input_dir, folder)
        output_folder = os.path.join(output_dir, folder)
        os.makedirs(output_folder, exist_ok=True)  # 确保输出目录存在

        # 遍历每个类别文件夹
        for class_folder in os.listdir(input_folder):
            class_input_folder = os.path.join(input_folder, class_folder)
            class_output_folder = os.path.join(output_folder, class_folder)

            os.makedirs(class_output_folder, exist_ok=True)  # 确保类别文件夹存在

            # 遍历每个类别下的图像
            if os.path.isdir(class_input_folder):
                for img_name in tqdm.tqdm(os.listdir(class_input_folder)):
                    img_path = os.path.join(class_input_folder, img_name)
            
                    # 读取图像
                    img_i = cv2.imread(img_path, 0)
                    img_shape = img_i.shape

                    # 预处理图像
                    img_i = cv2.resize(img_i, (768, 768), interpolation=cv2.INTER_LINEAR)  # 调整图像大小
                    img_i = img_i / 255.0
                    img_i = torch.from_numpy(img_i).float().unsqueeze(0).unsqueeze(0).to(device=device)

                  

                    # 后处理
                    img_o = net_BTOA(img_i)
                    img_o = get(img_o, img_shape)
                    ret, img_o = cv2.threshold(img_o, 80, 255, cv2.THRESH_BINARY)
                    kernel_2 = np.ones((2, 2), dtype=np.uint8)
                    img_o = cv2.dilate(img_o, kernel_2, 2)

                    # 保存处理后的图像
                    output_img_path = os.path.join(class_output_folder, img_name)
                    cv2.imwrite(output_img_path, img_o)
                    print(f"Saved: {output_img_path}")

                    # 清理显存
                    torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Image Transformation from Topographic to Handwriting")
    parser.add_argument('--model', type=str, default='netG_t2m_11.pth', help="Path to the trained model")
    parser.add_argument('--img_i', type=str, default='金文拓片60', help="Directory containing original topographic images")
    parser.add_argument('--img_o', type=str, default='金文拓片60转手写体', help="Directory to save the converted handwriting images")

    args = parser.parse_args()

    # 加载模型
    net_BTOA = UNet1(1).to(device=torch.device("cuda"))
    model_params = torch.load(args.model)
    net_BTOA.load_state_dict(model_params, strict=False)

    # 确保输出目录存在
    os.makedirs(args.img_o, exist_ok=True)

    # 处理图像
    process_images(args.img_i, args.img_o, net_BTOA, device=torch.device("cuda"))
