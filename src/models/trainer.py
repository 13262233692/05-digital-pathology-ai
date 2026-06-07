import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Optional
from loguru import logger
import numpy as np

from .srgan import SRGAN, Generator


class SRDataset(Dataset):
    def __init__(
        self,
        hr_dir: str,
        patch_size: int = 96,
        scale_factor: int = 4,
        augment: bool = True
    ):
        self.hr_paths = sorted(list(Path(hr_dir).glob("*.png")) + list(Path(hr_dir).glob("*.jpg")))
        if len(self.hr_paths) == 0:
            self.hr_paths = sorted(list(Path(hr_dir).rglob("*.png")) + list(Path(hr_dir).rglob("*.jpg")))
        
        if len(self.hr_paths) == 0:
            raise ValueError(f"No images found in {hr_dir}")
        
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) if augment else transforms.Compose([
            transforms.CenterCrop(patch_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(patch_size // scale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hr_img = Image.open(self.hr_paths[idx]).convert("RGB")
        hr_tensor = self.hr_transform(hr_img)
        
        lr_tensor = self.lr_transform(hr_tensor)
        
        return lr_tensor, hr_tensor


class SRGANTrainer:
    def __init__(
        self,
        srgan: SRGAN,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr_g: float = 1e-4,
        lr_d: float = 1e-4,
        content_weight: float = 1.0,
        adversarial_weight: float = 1e-3,
        pixel_weight: float = 1e-2,
        save_dir: str = "./checkpoints"
    ):
        self.srgan = srgan
        self.device = srgan.device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.content_weight = content_weight
        self.adversarial_weight = adversarial_weight
        self.pixel_weight = pixel_weight
        
        self.optimizer_g = optim.Adam(self.srgan.generator.parameters(), lr=lr_g, betas=(0.9, 0.999))
        self.optimizer_d = optim.Adam(self.srgan.discriminator.parameters(), lr=lr_d, betas=(0.9, 0.999))
        
        self.scheduler_g = optim.lr_scheduler.StepLR(self.optimizer_g, step_size=100, gamma=0.5)
        self.scheduler_d = optim.lr_scheduler.StepLR(self.optimizer_d, step_size=100, gamma=0.5)
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_psnr = 0.0

    def train_epoch(self, epoch: int) -> dict:
        self.srgan.generator.train()
        self.srgan.discriminator.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_content_loss = 0.0
        total_adversarial_loss = 0.0
        total_pixel_loss = 0.0
        
        num_batches = len(self.train_loader)
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(self.train_loader):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            self.optimizer_d.zero_grad()
            
            sr_imgs = self.srgan.generator(lr_imgs)
            
            real_loss = self.srgan.compute_adversarial_loss(hr_imgs, real=True)
            fake_loss = self.srgan.compute_adversarial_loss(sr_imgs.detach(), real=False)
            d_loss = (real_loss + fake_loss) * 0.5
            
            d_loss.backward()
            self.optimizer_d.step()
            
            self.optimizer_g.zero_grad()
            
            content_loss = self.srgan.compute_content_loss(sr_imgs, hr_imgs)
            adversarial_loss = self.srgan.compute_adversarial_loss(sr_imgs, real=True)
            pixel_loss = self.srgan.compute_pixel_loss(sr_imgs, hr_imgs)
            
            g_loss = (self.content_weight * content_loss + 
                     self.adversarial_weight * adversarial_loss + 
                     self.pixel_weight * pixel_loss)
            
            g_loss.backward()
            self.optimizer_g.step()
            
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_content_loss += content_loss.item()
            total_adversarial_loss += adversarial_loss.item()
            total_pixel_loss += pixel_loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(
                    f"Epoch [{epoch}] Batch [{batch_idx+1}/{num_batches}] | "
                    f"G_loss: {g_loss.item():.4f} | D_loss: {d_loss.item():.4f}"
                )
        
        return {
            "g_loss": total_g_loss / num_batches,
            "d_loss": total_d_loss / num_batches,
            "content_loss": total_content_loss / num_batches,
            "adversarial_loss": total_adversarial_loss / num_batches,
            "pixel_loss": total_pixel_loss / num_batches
        }

    def validate(self) -> dict:
        if self.val_loader is None:
            return {}
        
        self.srgan.generator.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in self.val_loader:
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                sr_imgs = self.srgan.generator(lr_imgs)
                
                sr_imgs = (sr_imgs + 1) / 2
                hr_imgs = (hr_imgs + 1) / 2
                
                mse = torch.mean((sr_imgs - hr_imgs) ** 2, dim=[1, 2, 3])
                psnr = 10 * torch.log10(1.0 / mse)
                total_psnr += psnr.mean().item()
        
        avg_psnr = total_psnr / num_batches
        
        return {"psnr": avg_psnr}

    def train(self, num_epochs: int = 200):
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")
        
        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            val_metrics = self.validate()
            
            log_msg = f"Epoch [{epoch}/{num_epochs}] | "
            log_msg += " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
            if val_metrics:
                log_msg += " | " + " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            logger.info(log_msg)
            
            if val_metrics and val_metrics.get("psnr", 0) > self.best_psnr:
                self.best_psnr = val_metrics["psnr"]
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % 10 == 0:
                self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.srgan.generator.state_dict(),
            "discriminator_state_dict": self.srgan.discriminator.state_dict(),
            "optimizer_g_state_dict": self.optimizer_g.state_dict(),
            "optimizer_d_state_dict": self.optimizer_d.state_dict(),
            "best_psnr": self.best_psnr
        }
        
        suffix = "best" if is_best else f"epoch_{epoch}"
        path = self.save_dir / f"srgan_{suffix}.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
        self.save_generator_for_triton(path)

    def save_generator_for_triton(self, checkpoint_path: str, output_path: Optional[str] = None):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        generator = Generator(
            in_channels=3,
            num_filters=64,
            num_residual_blocks=16,
            scale_factor=4
        )
        generator.load_state_dict(checkpoint["generator_state_dict"])
        generator.eval()
        
        if output_path is None:
            output_path = Path(checkpoint_path).parent / "generator.pt"
        
        traced_model = torch.jit.trace(
            generator,
            torch.randn(1, 3, 128, 128)
        )
        torch.jit.save(traced_model, output_path)
        logger.info(f"Saved traced generator for Triton: {output_path}")
        
        return output_path
