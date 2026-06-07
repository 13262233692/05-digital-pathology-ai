import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_stitcher.gaussian_stitcher import GaussianStitcher
from src.wsi_processor.tile_extractor import TileInfo


def test_gaussian_stitcher_basic():
    effective = (512 - 64) * 4
    num_tiles = 2
    output_size = (effective * num_tiles, effective * num_tiles)
    
    stitcher = GaussianStitcher(
        output_size=output_size,
        tile_size=512,
        overlap=32,
        scale_factor=4,
        blending_sigma=64.0,
    )
    
    sr_tile_size = 512 * 4
    
    for row in range(num_tiles):
        for col in range(num_tiles):
            tile_info = TileInfo(
                tile_id=f"tile_r{row}_c{col}",
                wsi_path="",
                x=0, y=0, width=512, height=512, level=0,
                row=row,
                col=col,
                total_rows=num_tiles,
                total_cols=num_tiles,
            )
            
            tile = np.ones((sr_tile_size, sr_tile_size, 3), dtype=np.uint8) * 128
            tile[:, :, 0] = np.clip(128 + (row * 20), 0, 255)
            tile[:, :, 1] = np.clip(128 + (col * 20), 0, 255)
            
            stitcher.add_tile(tile_info, tile)
    
    result = stitcher.finalize()
    
    assert result.shape == (output_size[1], output_size[0], 3), f"Expected {(output_size[1], output_size[0], 3)}, got {result.shape}"
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
    
    print("✓ Gaussian stitcher basic test passed")
    print(f"  Output shape: {result.shape}")
    print(f"  Value range: [{result.min()}, {result.max()}]")


def test_memory_estimation():
    mem_info = GaussianStitcher.estimate_memory_usage(4096, 4096, 3)
    
    assert "output_image_mb" in mem_info
    assert "weight_map_mb" in mem_info
    assert "estimated_total_mb" in mem_info
    
    print("✓ Memory estimation test passed")
    print(f"  Estimated total: {mem_info['estimated_total_mb']:.1f} MB")


if __name__ == "__main__":
    test_gaussian_stitcher_basic()
    test_memory_estimation()
    print("\nAll tests passed!")
