import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from openslide import OpenSlide

try:
    import openslide
    from openslide import OpenSlide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    OpenSlide = None
    logger.warning("OpenSlide not available. WSI reading will be limited.")


class WSIReader:
    def __init__(self, wsi_path: str, target_mpp: float = 0.25):
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("OpenSlide is required for WSI reading")
        
        self.wsi_path = Path(wsi_path)
        if not self.wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")
        
        self.target_mpp = target_mpp
        self._slide: Optional[OpenSlide] = None
        self._open()

    def _open(self) -> None:
        try:
            self._slide = OpenSlide(str(self.wsi_path))
            logger.info(f"Opened WSI: {self.wsi_path.name}")
            logger.info(f"Dimensions: {self.dimensions}")
            logger.info(f"Levels: {self.level_count}")
            logger.info(f"Level dimensions: {self.level_dimensions}")
        except Exception as e:
            logger.error(f"Failed to open WSI: {e}")
            raise

    @property
    def slide(self) -> OpenSlide:
        if self._slide is None:
            self._open()
        return self._slide

    @property
    def dimensions(self) -> Tuple[int, int]:
        return self.slide.dimensions

    @property
    def level_count(self) -> int:
        return self.slide.level_count

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        return self.slide.level_dimensions

    @property
    def level_downsamples(self) -> List[float]:
        return self.slide.level_downsamples

    def get_best_level_for_downsample(self, downsample: float) -> int:
        return self.slide.get_best_level_for_downsample(downsample)

    def get_mpp(self) -> Optional[Tuple[float, float]]:
        try:
            mpp_x = float(self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0))
            mpp_y = float(self.slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0))
            return (mpp_x, mpp_y) if mpp_x > 0 and mpp_y > 0 else None
        except (KeyError, ValueError):
            return None

    def get_scaling_factor(self) -> float:
        mpp = self.get_mpp()
        if mpp is None:
            logger.warning("MPP not found in WSI metadata, using default scaling factor 1.0")
            return 1.0
        current_mpp = (mpp[0] + mpp[1]) / 2
        return current_mpp / self.target_mpp

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int]
    ) -> np.ndarray:
        x, y = location
        w, h = size
        
        region = self.slide.read_region((x, y), level, (w, h))
        region_rgb = region.convert("RGB")
        return np.array(region_rgb, dtype=np.uint8)

    def read_thumbnail(self, max_size: Tuple[int, int] = (1024, 1024)) -> np.ndarray:
        thumbnail = self.slide.get_thumbnail(max_size)
        return np.array(thumbnail.convert("RGB"), dtype=np.uint8)

    def detect_tissue_mask(self, level: int = -1, threshold: int = 220) -> np.ndarray:
        if level < 0:
            level = self.level_count - 1
        
        level_dim = self.level_dimensions[level]
        thumbnail = self.read_region((0, 0), level, level_dim)
        
        gray = np.mean(thumbnail, axis=2)
        mask = gray < threshold
        
        from scipy.ndimage import binary_fill_holes, binary_opening
        mask = binary_opening(mask, iterations=2)
        mask = binary_fill_holes(mask)
        
        return mask.astype(np.uint8) * 255

    def get_tissue_bounding_box(self, mask: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]:
        if mask is None:
            mask = self.detect_tissue_mask()
        
        mask_binary = mask > 0
        rows = np.any(mask_binary, axis=1)
        cols = np.any(mask_binary, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return (0, 0, self.dimensions[0], self.dimensions[1])
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        downsample = self.level_downsamples[-1]
        x = int(cmin * downsample)
        y = int(rmin * downsample)
        w = int((cmax - cmin) * downsample)
        h = int((rmax - rmin) * downsample)
        
        return (x, y, w, h)

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
            self._slide = None
            logger.info(f"Closed WSI: {self.wsi_path.name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
