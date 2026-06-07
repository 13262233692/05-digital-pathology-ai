#!/usr/bin/env python
"""
SRGAN 模型训练脚本
使用方法: python scripts/train_model.py --train_dir /path/to/hr/train --val_dir /path/to/hr/val
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader
from loguru import logger

from src.models.srgan import SRGAN
from src.models.trainer import SRGANTrainer, SRDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train SRGAN model for Pathology Images")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to training HR images")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation HR images")
    parser.add_argument("--patch_size", type=int, default=96, help="HR patch size for training")
    parser.add_argument("--scale_factor", type=int, default=4, help="Super resolution scale factor")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="Generator learning rate")
    parser.add_argument("--lr_d", type=float, default=1e-4, help="Discriminator learning rate")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Checkpoint save directory")
    parser.add_argument("--num_workers", type=int, default=4, help="Data loader workers")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    train_dataset = SRDataset(
        hr_dir=args.train_dir,
        patch_size=args.patch_size,
        scale_factor=args.scale_factor,
        augment=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if args.val_dir:
        val_dataset = SRDataset(
            hr_dir=args.val_dir,
            patch_size=args.patch_size,
            scale_factor=args.scale_factor,
            augment=False
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    if val_loader:
        logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    srgan = SRGAN.build(
        in_channels=3,
        num_filters=64,
        num_residual_blocks=16,
        scale_factor=args.scale_factor,
        device=device
    )
    
    trainer = SRGANTrainer(
        srgan=srgan,
        train_loader=train_loader,
        val_loader=val_loader,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        save_dir=args.save_dir
    )
    
    trainer.train(num_epochs=args.num_epochs)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
