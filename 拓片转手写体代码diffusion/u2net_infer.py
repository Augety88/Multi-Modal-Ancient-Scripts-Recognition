import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import requests

# U^2-Net模型结构（简化版，适用于推理）
class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1*dirate, dilation=1*dirate)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class U2NET(nn.Module):
    # 省略详细结构，直接加载官方权重推理
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NET, self).__init__()
        # ...官方结构省略，推理用权重加载
    def forward(self, x):
        # ...推理用
        pass

def download_u2net_model(model_path):
    url = 'https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth'
    print('Downloading U^2-Net weights...')
    r = requests.get(url, stream=True)
    with open(model_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print('Download complete!')

def load_u2net(model_path):
    from 垃圾文件.u2net import U2NET as U2NET_full
    net = U2NET_full(3,1)
    # 兼容PyTorch 2.6+，强制weights_only=False
    net.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
    net.eval()
    return net

def u2net_infer(image, model, device='cpu'):
    # 输入PIL，输出二值mask（np.uint8, 0/255）
    im = image.convert('RGB').resize((320,320))
    im_np = np.array(im).astype(np.float32)/255.0
    im_np = im_np.transpose((2,0,1))[None]
    im_tensor = torch.from_numpy(im_np).to(device)
    with torch.no_grad():
        d1, *_ = model(im_tensor)
        pred = d1.squeeze().cpu().numpy()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        mask = (pred > 0.5).astype(np.uint8) * 255
    mask = Image.fromarray(mask).resize(image.size)
    return mask

if __name__ == '__main__':
    # 用法示例
    model_path = 'u2net.pth'
    if not os.path.exists(model_path):
        download_u2net_model(model_path)
    from 垃圾文件.u2net import U2NET as U2NET_full
    model = U2NET_full(3,1)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    img = Image.open('test.jpg')
    mask = u2net_infer(img, model)
    mask.save('test_mask.png')
