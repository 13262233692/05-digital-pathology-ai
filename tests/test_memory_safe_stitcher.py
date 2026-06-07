import sys
import numpy as np
from pathlib import Path
import gc
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.image_stitcher.memory_safe_stitcher import (
    MemorySafeGaussianStitcher,
    GPUMemoryMonitor,
    RingBuffer,
    TileBatchAggregator,
)
from src.wsi_processor.tile_extractor import TileInfo


def test_gpu_monitor():
    monitor = GPUMemoryMonitor(device_id=0, safety_threshold=0.85)
    
    usage = monitor.get_memory_usage()
    print(f"✓ GPU Monitor initialized")
    print(f"  Allocated: {usage['allocated_mb']:.1f} MB")
    print(f"  Reserved: {usage['reserved_mb']:.1f} MB")
    print(f"  Total: {usage['total_mb']:.1f} MB")
    print(f"  Free: {usage['free_mb']:.1f} MB")
    print(f"  Is safe: {monitor.is_safe()}")


def test_ring_buffer_cpu():
    shape = (256, 256, 3)
    ring = RingBuffer(buffer_shape=shape, num_buffers=4, use_gpu=False)
    
    idx1, buf1 = ring.acquire()
    assert buf1.shape == shape
    assert idx1 == 0
    
    idx2, buf2 = ring.acquire()
    assert idx2 == 1
    
    ring.release(idx1)
    
    remaining = list(ring._available)
    assert 0 in remaining
    
    idx3, buf3 = ring.acquire()
    assert buf3.shape == shape
    assert idx3 == 0
    
    ring.release(idx2)
    ring.release(idx3)
    
    assert len(ring._available) == 4
    
    print("✓ RingBuffer CPU test passed")


def test_ring_buffer_exhaustion():
    shape = (256, 256, 3)
    ring = RingBuffer(buffer_shape=shape, num_buffers=2, use_gpu=False)
    
    idx1, _ = ring.acquire()
    idx2, _ = ring.acquire()
    
    try:
        ring.acquire()
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "exhausted" in str(e)
    
    ring.release(idx1)
    idx3, _ = ring.acquire()
    assert idx3 == 0
    
    ring.release(idx2)
    ring.release(idx3)
    
    print("✓ RingBuffer exhaustion test passed")


def test_memory_safe_stitcher_cpu():
    sr_effective = (512 - 64) * 4
    num_cols = 3
    num_rows = 4
    output_w = sr_effective * num_cols
    output_h = sr_effective * num_rows
    
    stitcher = MemorySafeGaussianStitcher(
        output_size=(output_w, output_h),
        tile_size=512,
        overlap=32,
        scale_factor=4,
        blending_sigma=64.0,
        use_gpu=False,
        strip_height_factor=2,
        enable_disk_spill=False,
    )
    
    sr_tile_size = 512 * 4
    
    for row in range(num_rows):
        for col in range(num_cols):
            tile_info = TileInfo(
                tile_id=f"tile_r{row}_c{col}",
                wsi_path="",
                x=0, y=0, width=512, height=512, level=0,
                row=row,
                col=col,
                total_rows=num_rows,
                total_cols=num_cols,
            )
            
            tile = np.ones((sr_tile_size, sr_tile_size, 3), dtype=np.uint8) * 128
            tile[:, :, 0] = np.clip(100 + row * 20, 0, 255)
            tile[:, :, 1] = np.clip(100 + col * 20, 0, 255)
            
            stitcher.add_tile(tile_info, tile)
    
    result = stitcher.finalize()
    
    assert result.shape == (output_h, output_w, 3), f"Expected ({output_h}, {output_w}, 3), got {result.shape}"
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255
    
    print("✓ MemorySafeGaussianStitcher CPU test passed")
    print(f"  Output shape: {result.shape}")
    print(f"  Value range: [{result.min()}, {result.max()}]")


def test_memory_safe_stitcher_oom_fallback():
    print("\n--- OOM Fallback Test ---")
    
    sr_effective = (512 - 64) * 4
    num_cols = 2
    num_rows = 2
    output_w = sr_effective * num_cols
    output_h = sr_effective * num_rows
    
    stitcher = MemorySafeGaussianStitcher(
        output_size=(output_w, output_h),
        tile_size=512,
        overlap=32,
        scale_factor=4,
        blending_sigma=64.0,
        use_gpu=True,
        safety_threshold=0.01,
        strip_height_factor=1,
        enable_disk_spill=False,
    )
    
    sr_tile_size = 512 * 4
    
    for row in range(num_rows):
        for col in range(num_cols):
            tile_info = TileInfo(
                tile_id=f"tile_r{row}_c{col}",
                wsi_path="",
                x=0, y=0, width=512, height=512, level=0,
                row=row,
                col=col,
                total_rows=num_rows,
                total_cols=num_cols,
            )
            
            tile = np.ones((sr_tile_size, sr_tile_size, 3), dtype=np.uint8) * 128
            stitcher.add_tile(tile_info, tile)
    
    result = stitcher.finalize()
    
    assert result.shape == (output_h, output_w, 3)
    assert result.dtype == np.uint8
    
    print("✓ OOM Fallback test passed (CPU fallback works)")


def test_large_image_memory_estimation():
    print("\n--- Memory Estimation for Large WSI ---")
    
    sizes = [
        (20000, 20000, "Small WSI (20k x 20k)"),
        (50000, 50000, "Medium WSI (50k x 50k)"),
        (100000, 100000, "Large WSI (100k x 100k)"),
    ]
    
    for w, h, desc in sizes:
        old_est = {
            "total_mb": (w * h * 3 * 4 + w * h * 4 + 50 * 1024 * 1024) / (1024 * 1024)
        }
        
        new_est_cpu = MemorySafeGaussianStitcher.estimate_memory_usage(
            w, h, 3, use_gpu=False, strip_height=4096
        )
        new_est_gpu = MemorySafeGaussianStitcher.estimate_memory_usage(
            w, h, 3, use_gpu=True, strip_height=4096
        )
        
        print(f"\n  {desc}:")
        print(f"    Old approach (full GPU alloc): ~{old_est['total_mb'] / 1024:.1f} GB")
        print(f"    New approach CPU: {new_est_cpu['cpu_mb'] / 1024:.2f} GB")
        print(f"    New approach Peak GPU: {new_est_gpu['peak_gpu_mb']:.1f} MB")
        print(f"    GPU reduction: {(1 - new_est_gpu['peak_gpu_mb'] / max(1, old_est['total_mb'])) * 100:.1f}%")


def test_progress_tracking():
    sr_effective = (512 - 64) * 4
    output_w = sr_effective * 2
    output_h = sr_effective * 2
    
    stitcher = MemorySafeGaussianStitcher(
        output_size=(output_w, output_h),
        tile_size=512,
        overlap=32,
        scale_factor=4,
        use_gpu=False,
        strip_height_factor=1,
    )
    
    sr_tile_size = 512 * 4
    
    for row in range(2):
        for col in range(2):
            tile_info = TileInfo(
                tile_id=f"tile_r{row}_c{col}",
                wsi_path="",
                x=0, y=0, width=512, height=512, level=0,
                row=row,
                col=col,
                total_rows=2,
                total_cols=2,
            )
            tile = np.ones((sr_tile_size, sr_tile_size, 3), dtype=np.uint8) * 100
            stitcher.add_tile(tile_info, tile)
            
            progress = stitcher.get_progress()
            print(f"  After tile r{row}_c{col}: {progress['progress_pct']:.1f}%")
    
    result = stitcher.finalize()
    progress = stitcher.get_progress()
    assert progress['progress_pct'] == 100.0
    
    print("✓ Progress tracking test passed")


if __name__ == "__main__":
    test_gpu_monitor()
    test_ring_buffer_cpu()
    test_ring_buffer_exhaustion()
    test_memory_safe_stitcher_cpu()
    test_memory_safe_stitcher_oom_fallback()
    test_large_image_memory_estimation()
    test_progress_tracking()
    print("\n✅ All memory-safe stitcher tests passed!")
