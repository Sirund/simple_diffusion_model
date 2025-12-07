import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
# Import AMP untuk Mixed Precision (Hemat Memori T4)
from torch.cuda.amp import GradScaler, autocast

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        
        # Handle DataParallel wrapper during sampling
        if isinstance(model, nn.DataParallel):
            model_to_sample = model.module
        else:
            model_to_sample = model
            
        model_to_sample.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model_to_sample(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model_to_sample.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    
    # Inisialisasi Model
    model = UNet().to(device)
    
    # --- MODIFIKASI 1: MULTI-GPU SUPPORT ---
    if torch.cuda.device_count() > 1:
        print(f"🚀 Detected {torch.cuda.device_count()} GPUs! Activating DataParallel...")
        model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)
    
    # --- MODIFIKASI 2: MIXED PRECISION SCALER ---
    scaler = GradScaler()

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            # --- MODIFIKASI 3: MIXED PRECISION TRAINING LOOP ---
            optimizer.zero_grad()
            
            with autocast(): # Otomatis menggunakan FP16 untuk hemat memori
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)
            
            # Backward pass dengan Scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        # --- MODIFIKASI 4: SAMPLING HANYA TIAP 5 EPOCH ---
        # Ini mencegah bottleneck waktu dan VRAM
        if epoch % 5 == 0:
            sampled_images = diffusion.sample(model, n=images.shape[0])
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            
            # Save checkpoint (Handle DataParallel saving)
            save_path = os.path.join("models", args.run_name, f"ckpt.pt")
            if isinstance(model, nn.DataParallel):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)

def launch():
    import argparse
    parser = argparse.ArgumentParser()
    # Menambahkan argumen agar bisa diubah lewat command line
    parser.add_argument('--run_name', type=str, default="DDPM_Uncondtional")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32) # Batch size aman untuk T4
    parser.add_argument('--image_size', type=int, default=64)
    # Default path diarahkan ke input, tapi user HARUS mengubahnya lewat argumen jika struktur salah
    parser.add_argument('--dataset_path', type=str, default=r"/kaggle/input/") 
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--lr', type=float, default=3e-4)
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    launch()
