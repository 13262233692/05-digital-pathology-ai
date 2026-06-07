import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from loguru import logger
import tifffile
from datetime import datetime


class OME_TIFFWriter:
    def __init__(
        self,
        output_path: str,
        image_shape: Tuple[int, int],
        num_channels: int = 3,
        pixel_size: float = 0.25,
        compression: str = "lzw",
        tile_size: int = 1024,
        software: str = "PathologySR"
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.height, self.width = image_shape
        self.num_channels = num_channels
        self.pixel_size = pixel_size
        self.compression = compression
        self.tile_size = tile_size
        self.software = software
        
        self._writer: Optional[tifffile.TiffWriter] = None

    def _create_ome_metadata(self) -> Dict:
        from ome_types import from_tiff, OME, Image, Pixels, Channel, Plane
        
        ome = OME()
        
        image = Image(
            id="Image:0",
            name=self.output_path.stem,
            acquisition_date=datetime.now().isoformat()
        )
        
        pixels = Pixels(
            id="Pixels:0",
            dimension_order="XYCZT",
            type="uint8",
            size_x=self.width,
            size_y=self.height,
            size_c=self.num_channels,
            size_z=1,
            size_t=1,
            physical_size_x=self.pixel_size,
            physical_size_y=self.pixel_size,
            physical_size_z=1.0,
        )
        
        for c in range(self.num_channels):
            channel_names = ["Red", "Green", "Blue"]
            channel = Channel(
                id=f"Channel:{c}",
                name=channel_names[c] if c < 3 else f"Channel{c}",
                samples_per_pixel=1
            )
            pixels.channels.append(channel)
        
        plane = Plane(
            the_c=0,
            the_z=0,
            the_t=0,
            position_x=0.0,
            position_y=0.0,
            position_z=0.0
        )
        pixels.planes.append(plane)
        
        image.pixels = pixels
        ome.images.append(image)
        
        return ome

    def write(self, image: np.ndarray) -> None:
        logger.info(f"Writing OME-TIFF to {self.output_path}")
        logger.info(f"Image shape: {image.shape}, dtype: {image.dtype}")
        
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        
        if image.shape[2] != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} channels, got {image.shape[2]}")
        
        if image.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Expected shape ({self.height}, {self.width}), "
                f"got ({image.shape[0]}, {image.shape[1]})"
            )
        
        try:
            ome_metadata = self._create_ome_metadata()
            
            with tifffile.TiffWriter(self.output_path, ome=True) as tif:
                tif.write(
                    image,
                    photometric='rgb' if self.num_channels == 3 else 'minisblack',
                    compression=self.compression,
                    tile=(self.tile_size, self.tile_size),
                    metadata={
                        'axes': 'YXC',
                        'software': self.software,
                    }
                )
            
            logger.info(f"Successfully wrote OME-TIFF: {self.output_path}")
            file_size_mb = self.output_path.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
        except ImportError:
            logger.warning("ome-types not available, writing basic TIFF")
            self._write_basic_tiff(image)
        except Exception as e:
            logger.error(f"Error writing OME-TIFF: {e}")
            raise

    def _write_basic_tiff(self, image: np.ndarray) -> None:
        with tifffile.TiffWriter(self.output_path) as tif:
            tif.write(
                image,
                photometric='rgb' if self.num_channels == 3 else 'minisblack',
                compression=self.compression,
                tile=(self.tile_size, self.tile_size),
                metadata={'software': self.software}
            )

    def write_pyramid(self, image: np.ndarray, num_levels: int = 4) -> None:
        logger.info(f"Writing pyramidal OME-TIFF with {num_levels} levels")
        
        import cv2
        
        levels = [image]
        current = image
        
        for _ in range(num_levels - 1):
            h, w = current.shape[:2]
            current = cv2.resize(
                current,
                (w // 2, h // 2),
                interpolation=cv2.INTER_AREA
            )
            levels.append(current)
        
        try:
            ome_metadata = self._create_ome_metadata()
            
            with tifffile.TiffWriter(self.output_path, ome=True, bigtiff=True) as tif:
                for i, level in enumerate(levels):
                    is_last = (i == len(levels) - 1)
                    tif.write(
                        level,
                        photometric='rgb' if self.num_channels == 3 else 'minisblack',
                        compression=self.compression,
                        tile=(self.tile_size, self.tile_size),
                        subfiletype=1 if not is_last else 0,
                        metadata={
                            'axes': 'YXC',
                            'software': self.software,
                        }
                    )
            
            logger.info(f"Successfully wrote pyramidal OME-TIFF")
            
        except Exception as e:
            logger.error(f"Error writing pyramidal OME-TIFF: {e}")
            raise

    @staticmethod
    def estimate_file_size(
        width: int,
        height: int,
        num_channels: int = 3,
        compression_ratio: float = 2.0
    ) -> float:
        raw_size = width * height * num_channels
        compressed_size = raw_size / compression_ratio
        return compressed_size / (1024 * 1024 * 1024)
