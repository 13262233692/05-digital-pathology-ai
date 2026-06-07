import numpy as np
from typing import List, Tuple, Iterator, Optional
from dataclasses import dataclass
from loguru import logger
from pathlib import Path

from .wsi_reader import WSIReader


@dataclass
class TileInfo:
    tile_id: str
    wsi_path: str
    x: int
    y: int
    width: int
    height: int
    level: int
    row: int
    col: int
    total_rows: int
    total_cols: int


class TileExtractor:
    def __init__(
        self,
        wsi_reader: WSIReader,
        tile_size: int = 512,
        overlap: int = 32,
        level: int = 0,
        use_tissue_mask: bool = True
    ):
        self.wsi_reader = wsi_reader
        self.tile_size = tile_size
        self.overlap = overlap
        self.level = level
        self.use_tissue_mask = use_tissue_mask
        
        self.effective_tile_size = tile_size - 2 * overlap
        if self.effective_tile_size <= 0:
            raise ValueError(f"Overlap ({overlap}) too large for tile size ({tile_size})")
        
        self._bbox: Optional[Tuple[int, int, int, int]] = None
        self._mask: Optional[np.ndarray] = None
        self._grid: Optional[List[TileInfo]] = None

    def _get_tissue_bbox(self) -> Tuple[int, int, int, int]:
        if self._bbox is None:
            if self.use_tissue_mask:
                self._mask = self.wsi_reader.detect_tissue_mask()
                self._bbox = self.wsi_reader.get_tissue_bounding_box(self._mask)
            else:
                w, h = self.wsi_reader.dimensions
                self._bbox = (0, 0, w, h)
        return self._bbox

    def _compute_grid(self) -> List[TileInfo]:
        if self._grid is not None:
            return self._grid
        
        x_start, y_start, w, h = self._get_tissue_bbox()
        x_end = x_start + w
        y_end = y_start + h
        
        step = self.effective_tile_size
        
        cols = max(1, int(np.ceil((x_end - x_start - 2 * self.overlap) / step)))
        rows = max(1, int(np.ceil((y_end - y_start - 2 * self.overlap) / step)))
        
        logger.info(f"Computing tile grid: {rows} rows x {cols} cols = {rows*cols} tiles")
        
        tiles = []
        wsi_name = Path(self.wsi_reader.wsi_path).stem
        
        for row in range(rows):
            for col in range(cols):
                x = x_start + col * step
                y = y_start + row * step
                
                x = max(x_start, x - self.overlap)
                y = max(y_start, y - self.overlap)
                
                tile_w = min(self.tile_size, x_end - x)
                tile_h = min(self.tile_size, y_end - y)
                
                tile_id = f"{wsi_name}_r{row:04d}_c{col:04d}"
                
                tile_info = TileInfo(
                    tile_id=tile_id,
                    wsi_path=str(self.wsi_reader.wsi_path),
                    x=x,
                    y=y,
                    width=tile_w,
                    height=tile_h,
                    level=self.level,
                    row=row,
                    col=col,
                    total_rows=rows,
                    total_cols=cols
                )
                
                if self._is_valid_tile(tile_info):
                    tiles.append(tile_info)
        
        logger.info(f"Generated {len(tiles)} valid tiles after tissue filtering")
        self._grid = tiles
        return tiles

    def _is_valid_tile(self, tile_info: TileInfo) -> bool:
        if not self.use_tissue_mask or self._mask is None:
            return True
        
        mask_level = self.wsi_reader.level_count - 1
        downsample = self.wsi_reader.level_downsamples[mask_level]
        
        mask_h, mask_w = self._mask.shape
        
        mx_start = max(0, int(tile_info.x / downsample))
        my_start = max(0, int(tile_info.y / downsample))
        mx_end = min(mask_w, int((tile_info.x + tile_info.width) / downsample))
        my_end = min(mask_h, int((tile_info.y + tile_info.height) / downsample))
        
        if mx_end <= mx_start or my_end <= my_start:
            return False
        
        mask_patch = self._mask[my_start:my_end, mx_start:mx_end]
        tissue_ratio = np.sum(mask_patch > 0) / mask_patch.size
        
        return tissue_ratio > 0.1

    def extract_tile(self, tile_info: TileInfo) -> np.ndarray:
        region = self.wsi_reader.read_region(
            location=(tile_info.x, tile_info.y),
            level=tile_info.level,
            size=(tile_info.width, tile_info.height)
        )
        
        if region.shape[0] < self.tile_size or region.shape[1] < self.tile_size:
            padded = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
            padded[:region.shape[0], :region.shape[1], :] = region
            region = padded
        
        return region

    def extract_all_tiles(self) -> Iterator[Tuple[TileInfo, np.ndarray]]:
        grid = self._compute_grid()
        for tile_info in grid:
            tile_data = self.extract_tile(tile_info)
            yield tile_info, tile_data

    def get_tile_grid(self) -> List[TileInfo]:
        return self._compute_grid()

    def get_output_dimensions(self) -> Tuple[int, int]:
        x_start, y_start, w, h = self._get_tissue_bbox()
        scale_factor = 4
        out_w = (w + self.effective_tile_size - 1) // self.effective_tile_size * self.effective_tile_size * scale_factor
        out_h = (h + self.effective_tile_size - 1) // self.effective_tile_size * self.effective_tile_size * scale_factor
        return out_w, out_h

    def get_output_origin(self) -> Tuple[int, int]:
        x_start, y_start, _, _ = self._get_tissue_bbox()
        return x_start, y_start
