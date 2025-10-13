import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from models_cvae import cVAE
from tqdm import tqdm

class PairedDataset(Dataset):
    def __init__(self, dir_A, dir_B, img_size=64):
        self.A_list = sorted([os.path.join(dir_A, f) for f in os.listdir(dir_A) if f.endswith(('.png', '.jpg', '.bmp'))])
        self.B_list = sorted([os.path.join(dir_B, f) for f in os.listdir(dir_B) if f.endswith(('.png', '.jpg', '.bmp'))])
        self.img_size = img_size
    def __len__(self):
        return min(len(self.A_list), len(self.B_list))
    def __getitem__(self, idx):
        img_A = cv2.imread(self.A_list[idx], 0)
        img_B = cv2.imread(self.B_list[idx], 0)
        img_A = cv2.resize(img_A, (self.img_size, self.img_size)) / 255.0
        img_B = cv2.resize(img_B, (self.img_size, self.img_size)) / 255.0
        img_A = torch.from_numpy(img_A).float().unsqueeze(0)
        img_B = torch.from_numpy(img_B).float().unsqueeze(0)
        return img_A, img_B

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld

def train_cvae(dir_A, dir_B, save_path, epochs=50, batch_size=2, lr=1e-4, img_size=64, latent_dim=128, device='cuda'):
    dataset = PairedDataset(dir_A, dir_B, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = cVAE(1, 1, latent_dim, img_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for img_A, img_B in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            img_A, img_B = img_A.to(device), img_B.to(device)
            out, mu, logvar = model(img_A)
            loss, _, _ = vae_loss(out, img_B, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}: loss={total_loss/len(dataloader):.4f}')
        if (epoch+1) % 10 == 0 or epoch == epochs-1:
            torch.save(model.state_dict(), save_path)
    print('训练完成，模型已保存到', save_path)

    # ====== 测试集推理与评估 ======
    test_A_dir = dir_A.replace('train', 'test')
    test_B_dir = dir_B.replace('train', 'test')
    if os.path.exists(test_A_dir) and os.path.exists(test_B_dir):
        test_dataset = PairedDataset(test_A_dir, test_B_dir, img_size)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        model.eval()
        os.makedirs('output_test', exist_ok=True)
        total_mse = 0
        with torch.no_grad():
            for i, (img_A, img_B) in enumerate(test_loader):
                img_A, img_B = img_A.to(device), img_B.to(device)
                out, _, _ = model(img_A)
                mse = nn.functional.mse_loss(out, img_B).item()
                total_mse += mse
                out_img = out.squeeze().cpu().numpy() * 255
                out_img = np.clip(out_img, 0, 255).astype('uint8')
                cv2.imwrite(f'output_test/{i:04d}.png', out_img)
        print(f'Test set average MSE: {total_mse/len(test_loader):.4f}')
    else:
        print('未检测到test/A和test/B，跳过测试集推理。')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_A', type=str, default='dataset/train/A', help='拓片图片文件夹')
    parser.add_argument('--dir_B', type=str, default='dataset/train/B', help='手写体图片文件夹')
    parser.add_argument('--save_path', type=str, default='cvae.pth', help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--latent_dim', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    train_cvae(args.dir_A, args.dir_B, args.save_path, args.epochs, args.batch_size, args.lr, args.img_size, args.latent_dim, args.device)
