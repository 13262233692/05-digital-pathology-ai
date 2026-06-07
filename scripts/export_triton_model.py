#!/usr/bin/env python
"""
导出训练好的模型到 Triton 模型仓库
使用方法: python scripts/export_triton_model.py --checkpoint ./checkpoints/srgan_best.pth --output ./triton_model_repository/srgan/1/model.pt
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from loguru import logger

from src.models.srgan import Generator


def parse_args():
    parser = argparse.ArgumentParser(description="Export SRGAN generator to TorchScript for Triton")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output TorchScript path")
    parser.add_argument("--num_residual_blocks", type=int, default=16, help="Number of residual blocks")
    parser.add_argument("--num_filters", type=int, default=64, help="Number of filters")
    parser.add_argument("--scale_factor", type=int, default=4, help="Scale factor")
    parser.add_argument("--input_size", type=int, default=512, help="Input tile size")
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    generator = Generator(
        in_channels=3,
        num_filters=args.num_filters,
        num_residual_blocks=args.num_residual_blocks,
        scale_factor=args.scale_factor
    )
    
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()
    
    logger.info("Tracing model with TorchScript...")
    example_input = torch.randn(1, 3, args.input_size, args.input_size)
    traced_model = torch.jit.trace(generator, example_input)
    
    logger.info("Saving TorchScript model...")
    torch.jit.save(traced_model, output_path)
    
    logger.info(f"Model exported successfully: {output_path}")
    logger.info(f"Input shape: [1, 3, {args.input_size}, {args.input_size}]")
    logger.info(f"Output shape: [1, 3, {args.input_size * args.scale_factor}, {args.input_size * args.scale_factor}]")


if __name__ == "__main__":
    main()
