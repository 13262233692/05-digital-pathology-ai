#!/usr/bin/env python
"""
单张 WSI 处理脚本 (不使用 Celery 的本地处理)
使用方法: python scripts/process_single_wsi.py --wsi_path /path/to/image.svs --output_dir ./output
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger
import yaml

from src.wsi_processor.wsi_reader import WSIReader
from src.wsi_processor.tile_extractor import TileExtractor
from src.triton_client.inference_client import TritonInferenceClient
from src.image_stitcher.memory_safe_stitcher import MemorySafeGaussianStitcher
from src.image_stitcher.ome_tiff_writer import OME_TIFFWriter


def load_config():
    config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Process single WSI with SR pipeline")
    parser.add_argument("--wsi_path", type=str, required=True, help="Path to WSI file (.svs)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--triton_url", type=str, default="localhost:8001", help="Triton server URL")
    parser.add_argument("--no_tissue_mask", action="store_true", help="Disable tissue mask filtering")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    
    wsi_path = Path(args.wsi_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not wsi_path.exists():
        logger.error(f"WSI file not found: {wsi_path}")
        sys.exit(1)
    
    logger.info(f"Processing WSI: {wsi_path}")
    
    with WSIReader(str(wsi_path), target_mpp=config["wsi"]["target_mpp"]) as reader:
        extractor = TileExtractor(
            reader,
            tile_size=config["wsi"]["tile_size"],
            overlap=config["wsi"]["overlap"],
            level=config["wsi"]["pyramid_level"],
            use_tissue_mask=not args.no_tissue_mask
        )
        
        tile_grid = extractor.get_tile_grid()
        total_tiles = len(tile_grid)
        
        logger.info(f"Total tiles: {total_tiles}")
        
        output_dims = extractor.get_output_dimensions()
        logger.info(f"Output dimensions (SR): {output_dims}")
        
        mem_est = MemorySafeGaussianStitcher.estimate_memory_usage(
            output_dims[0], output_dims[1]
        )
        logger.info(f"Estimated memory usage: CPU {mem_est['cpu_mb']:.1f} MB, Peak GPU {mem_est['peak_gpu_mb']:.1f} MB")
        
        stitch_config = config["stitching"]
        stitcher = MemorySafeGaussianStitcher(
            output_size=output_dims,
            tile_size=config["wsi"]["tile_size"],
            overlap=config["wsi"]["overlap"],
            scale_factor=config["srgan"]["scale_factor"],
            blending_sigma=stitch_config["blending_sigma"],
            use_gpu=stitch_config.get("use_gpu", True),
            safety_threshold=stitch_config.get("safety_threshold", 0.8),
            strip_height_factor=stitch_config.get("strip_height_factor", 16),
            enable_disk_spill=stitch_config.get("enable_disk_spill", True),
        )
        
        with TritonInferenceClient(
            server_url=args.triton_url,
            model_name=config["triton"]["model_name"],
            model_version=config["triton"]["model_version"],
        ) as triton_client:
            
            batch_size = args.batch_size
            
            for i in range(0, len(tile_grid), batch_size):
                batch_tiles_info = tile_grid[i:i + batch_size]
                
                batch_tiles = []
                for tile_info in batch_tiles_info:
                    tile_data = extractor.extract_tile(tile_info)
                    batch_tiles.append(tile_data)
                
                batch_array = np.stack(batch_tiles, axis=0)
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(tile_grid) + batch_size - 1)//batch_size}")
                
                sr_batch = triton_client.infer(batch_array)
                
                for j, tile_info in enumerate(batch_tiles_info):
                    stitcher.add_tile(tile_info, sr_batch[j])
                
                progress = min(100.0 * (i + len(batch_tiles_info)) / len(tile_grid), 100.0)
                logger.info(f"Progress: {progress:.1f}%")
    
    logger.info("All tiles processed, stitching...")
    final_image = stitcher.finalize()
    
    output_path = output_dir / f"{wsi_path.stem}_super_resolved.ome.tiff"
    
    writer = OME_TIFFWriter(
        output_path=str(output_path),
        image_shape=(final_image.shape[0], final_image.shape[1]),
        num_channels=final_image.shape[2],
        pixel_size=config["wsi"]["target_mpp"] / config["srgan"]["scale_factor"],
        compression=config["stitching"]["compression"],
    )
    writer.write(final_image)
    
    logger.info(f"Done! Output saved to: {output_path}")
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Output file size: {file_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
