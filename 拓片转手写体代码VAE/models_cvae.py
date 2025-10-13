import torch
import torch.nn as nn
import torch.nn.functional as F

class cVAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=128, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        # 编码器：输入为 拓片A
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * (img_size // 16) * (img_size // 16), latent_dim)
        self.fc_logvar = nn.Linear(256 * (img_size // 16) * (img_size // 16), latent_dim)
        # 解码器：输入为z+拓片A
        self.fc_decode = nn.Linear(latent_dim + in_channels * img_size * img_size, 256 * (img_size // 16) * (img_size // 16))
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, img_size // 16, img_size // 16)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1), nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        enc = self.encoder(x)
        mu = self.fc_mu(enc)
        logvar = self.fc_logvar(enc)
        z = self.reparameterize(mu, logvar)
        # 条件：将x展平后与z拼接
        x_flat = x.view(x.size(0), -1)
        z_cond = torch.cat([z, x_flat], dim=1)
        dec = self.fc_decode(z_cond)
        out = self.decoder(dec)
        return out, mu, logvar
