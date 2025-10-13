import torch
from 垃圾文件.models_adafunet import UNet1,Generator
from 垃圾文件.models import *
import cv2
import numpy as np
import os
import tqdm
import argparse

def get(image_A, img_shape):
    # image_A = (image_A / 2.0) + 0.5
    image_A = image_A * 255
    image_A = torch.squeeze(image_A)
    image_A = image_A.detach()
    image = np.array(image_A.cpu())
    image_A = np.uint8(image)
    image_A = cv2.resize(image_A, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    # np.transpose(image_A,(1,2,0))
    return image_A

def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


if __name__ == "__main__":
    torch.cuda.empty_cache()  # 清空显存
    print(torch.cuda.memory_allocated())  # 查看当前已分配的显存
    print(torch.cuda.memory_reserved())  # 查看当前保留的显存


    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--model', type=str, default='netG_t2m_11.pth')
    parser.add_argument('--img_i', type=str, default='input/')
    parser.add_argument('--img_o', type=str, default='output/')
    
    
    args = parser.parse_args()

    net_BTOA = UNet1(1).to(device=torch.device("cuda"))
    #net_BTOA = GeneratorUNet().to(device=torch.device("cuda"))
    modle_param = torch.load(args.model)
    net_BTOA.load_state_dict(modle_param, strict=False)
    os.makedirs(args.img_o, exist_ok=True)
    for img_name in tqdm.tqdm(os.listdir(args.img_i)[:200]):
        img_i = cv2.imread(os.path.join(args.img_i, img_name), 0)
        img_shape = img_i.shape
        img_i = cv2.resize(img_i, (800, 800), interpolation=cv2.INTER_LINEAR)
        img_i = img_i/255.0
        img_i = torch.from_numpy(img_i).float().unsqueeze(0).unsqueeze(0).to(device=torch.device("cuda"))
        img_o = net_BTOA(img_i)
        img_o = get(img_o, img_shape)
        ret,img_o = cv2.threshold(img_o,80,255,cv2.THRESH_BINARY)
        kernel_2 = np.ones((2, 2), dtype=np.uint8)
        img_o = cv2.dilate(img_o, kernel_2, 2)
        cv2.imwrite(os.path.join(args.img_o, img_name), img_o)
