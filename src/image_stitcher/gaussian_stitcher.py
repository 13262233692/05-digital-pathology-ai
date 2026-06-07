import numpy as np
from typing import List, Tuple, Dict, Optional
from loguru import logger
from pathlib import Path
import math

from ..wsi_processor.tile_extractor import TileInfo


class GaussianStitcher:
    def __init__(
        self,
        output_size: Tuple[int, int],
        tile_size: int = 512,
        overlap: int = 32,
        scale_factor: int = 4,
        blending_sigma: float = 64.0,
        num_channels: int = 3,
        dtype: np.dtype = np.uint8,
        use_gpu: bool = False
    ):
        self.output_width, self.output_height = output_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.scale_factor = scale_factor
        self.blending_sigma = blending_sigma
        self.num_channels = num_channels
        self.dtype = dtype
        self.use_gpu = use_gpu
        
        self.sr_tile_size = tile_size * scale_factor
        self.sr_overlap = overlap * scale_factor
        self.sr_effective = (tile_size - 2 * overlap) * scale_factor
        
        self._output_img: Optional[np.ndarray] = None
        self._weight_map: Optional[np.ndarray] = None
        self._gaussian_kernel: Optional[np.ndarray] = None
        
        self._initialize()

    def _initialize(self) -> None:
        logger.info(f"Initializing stitcher for output size: {self.output_width}x{self.output_height}")
        
        if self.use_gpu:
            try:
                import cupy as cp
                self.xp = cp
                logger.info("Using GPU acceleration (CuPy)")
            except ImportError:
                logger.warning("CuPy not available, falling back to CPU")
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np
        
        self._output_img = self.xp.zeros(
            (self.output_height, self.output_width, self.num_channels),
            dtype=np.float32
        )
        self._weight_map = self.xp.zeros(
            (self.output_height, self.output_width),
            dtype=np.float32
        )
        
        self._gaussian_kernel = self._create_gaussian_kernel()
        logger.info(f"Gaussian stitcher initialized")

    def _create_gaussian_kernel(self) -> np.ndarray:
        size = self.sr_tile_size
        sigma = self.blending_sigma
        
        center = size / 2.0
        
        y, x = np.ogrid[:size, :size]
        dist_sq = (x - center) ** 2 + (y - center) ** 2
        
        kernel = np.exp(-dist_sq / (2 * sigma ** 2))
        
        edge_dist = np.minimum(
            np.minimum(x, size - 1 - x),
            np.minimum(y, size - 1 - y)
        )
        
        edge_width = self.sr_overlap
        edge_mask = np.minimum(1.0, edge_dist / max(1, edge_width))
        edge_mask = np.sin(edge_mask * np.pi / 2) ** 2
        
        kernel = kernel * edge_mask
        
        kernel = kernel / kernel.max()
        
        if self.use_gpu:
            kernel = self.xp.asarray(kernel)
        
        return kernel

    def _get_tile_position(self, tile_info: TileInfo) -> Tuple[int, int, int, int]:
        out_x = tile_info.col * self.sr_effective
        out_y = tile_info.row * self.sr_effective
        
        tile_h = min(self.sr_tile_size, self.output_height - out_y)
        tile_w = min(self.sr_tile_size, self.output_width - out_x)
        
        return out_x, out_y, tile_w, tile_h

    def add_tile(self, tile_info: TileInfo, sr_tile: np.ndarray) -> None:
        if self._output_img is None or self._weight_map is None:
            raise RuntimeError("Stitcher not initialized")
        
        out_x, out_y, tile_w, tile_h = self._get_tile_position(tile_info)
        
        if self.use_gpu:
            sr_tile = self.xp.asarray(sr_tile)
        
        sr_tile = sr_tile.astype(np.float32)
        
        kernel = self._gaussian_kernel[:tile_h, :tile_w]
        kernel_3d = kernel[..., self.xp.newaxis]
        
        weighted_tile = sr_tile[:tile_h, :tile_w, :] * kernel_3d
        
        self._output_img[out_y:out_y + tile_h, out_x:out_x + tile_w, :] += weighted_tile
        self._weight_map[out_y:out_y + tile_h, out_x:out_x + tile_w] += kernel

    def add_tiles_batch(self, tiles: List[Tuple[TileInfo, np.ndarray]]) -> None:
        for tile_info, sr_tile in tiles:
            self.add_tile(tile_info, sr_tile)

    def finalize(self) -> np.ndarray:
        if self._output_img is None or self._weight_map is None:
            raise RuntimeError("Stitcher not initialized")
        
        logger.info("Finalizing stitching with Gaussian blending...")
        
        weight_map_3d = self._weight_map[..., self.xp.newaxis]
        weight_map_3d = self.xp.maximum(weight_map_3d, 1e-8)
        
        result = self._output_img / weight_map_3d
        
        result = self.xp.clip(result, 0, 255)
        
        if self.use_gpu:
            result = self.xp.asnumpy(result)
        
        result = result.astype(self.dtype)
        
        logger.info(f"Stitching complete, output shape: {result.shape}")
        
        return result

    def save_progress(self) -> dict:
        total_pixels = self.output_height * self.output_width
        filled_pixels = int(self.xp.sum(self._weight_map > 0))
        progress = filled_pixels / total_pixels * 100
        
        return {
            "progress": progress,
            "filled_pixels": filled_pixels,
            "total_pixels": total_pixels
        }

    def reset(self) -> None:
        self._output_img = None
        self._weight_map = None
        self._initialize()

    @staticmethod
    def estimate_memory_usage(
        width: int,
        height: int,
        num_channels: int = 3,
        use_gpu: bool = False
    ) -> dict:
        pixel_count = width * height
        output_mem = pixel_count * num_channels * 4
        weight_mem = pixel_count * 4
        total_mem = output_mem + weight_mem + (1024 * 1024 * 50)
        
        return {
            "output_image_mb": output_mem / (1024 * 1024),
            "weight_map_mb": weight_mem / (1024 * 1024),
            "estimated_total_mb": total_mem / (1024 * 1024)
        }
