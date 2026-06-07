import numpy as np
from typing import List, Tuple, Dict, Optional, Deque
from collections import deque
from loguru import logger
from pathlib import Path
import gc
import warnings
import tempfile
import os

from ..wsi_processor.tile_extractor import TileInfo

warnings.filterwarnings("ignore", category=UserWarning)


class GPUMemoryMonitor:
    def __init__(self, device_id: int = 0, safety_threshold: float = 0.85):
        self.device_id = device_id
        self.safety_threshold = safety_threshold
        self._torch_available = False
        self._cupy_available = False
        
        try:
            import torch
            self._torch_available = True
            self._torch = torch
        except ImportError:
            pass
        
        try:
            import cupy as cp
            self._cupy_available = True
            self._cp = cp
        except ImportError:
            pass

    def get_memory_usage(self) -> Dict[str, float]:
        usage = {"allocated_mb": 0.0, "reserved_mb": 0.0, "total_mb": 0.0, "free_mb": 0.0}
        
        if self._torch_available and self._torch.cuda.is_available():
            device = self._torch.device(f"cuda:{self.device_id}")
            usage["allocated_mb"] = self._torch.cuda.memory_allocated(device) / (1024 ** 2)
            usage["reserved_mb"] = self._torch.cuda.memory_reserved(device) / (1024 ** 2)
            props = self._torch.cuda.get_device_properties(device)
            usage["total_mb"] = props.total_memory / (1024 ** 2)
            usage["free_mb"] = usage["total_mb"] - usage["reserved_mb"]
        
        elif self._cupy_available:
            try:
                mempool = self._cp.get_default_memory_pool()
                usage["allocated_mb"] = mempool.used_bytes() / (1024 ** 2)
                usage["reserved_mb"] = mempool.total_bytes() / (1024 ** 2)
                device = self._cp.cuda.Device(self.device_id)
                mem_info = device.mem_info
                usage["total_mb"] = mem_info[1] / (1024 ** 2)
                usage["free_mb"] = mem_info[0] / (1024 ** 2)
            except:
                pass
        
        return usage

    def is_safe(self) -> bool:
        mem = self.get_memory_usage()
        if mem["total_mb"] > 0:
            usage_ratio = mem["reserved_mb"] / mem["total_mb"]
            return usage_ratio < self.safety_threshold
        return True

    def force_gc(self) -> None:
        if self._torch_available and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
            self._torch.cuda.synchronize()
        
        if self._cupy_available:
            try:
                mempool = self._cp.get_default_memory_pool()
                mempool.free_all_blocks()
            except:
                pass
        
        gc.collect()

    def get_available_memory_mb(self) -> float:
        mem = self.get_memory_usage()
        return mem["free_mb"]


class RingBuffer:
    def __init__(
        self,
        buffer_shape: Tuple[int, ...],
        num_buffers: int = 4,
        dtype=np.float32,
        use_gpu: bool = False,
        device_id: int = 0
    ):
        self.buffer_shape = buffer_shape
        self.num_buffers = num_buffers
        self.dtype = dtype
        self.use_gpu = use_gpu
        self.device_id = device_id
        
        self._buffers: Deque = deque()
        self._available: Deque[int] = deque()
        self._xp = None
        self._initialize()

    def _initialize(self) -> None:
        self._xp = np
        if self.use_gpu:
            try:
                import cupy as cp
                self._xp = cp
                with cp.cuda.Device(self.device_id):
                    for i in range(self.num_buffers):
                        buf = cp.zeros(self.buffer_shape, dtype=self.dtype)
                        self._buffers.append(buf)
                        self._available.append(i)
                logger.info(f"RingBuffer initialized on GPU:{self.device_id}, {self.num_buffers} x {self.buffer_shape}")
                return
            except ImportError:
                logger.warning("CuPy not available, falling back to CPU")
                self.use_gpu = False
                self._xp = np
            except Exception as e:
                logger.warning(f"GPU RingBuffer init failed: {e}, falling back to CPU")
                self.use_gpu = False
                self._xp = np
        
        for i in range(self.num_buffers):
            buf = np.zeros(self.buffer_shape, dtype=self.dtype)
            self._buffers.append(buf)
            self._available.append(i)
        
        logger.info(f"RingBuffer initialized on CPU, {self.num_buffers} x {self.buffer_shape}")

    def acquire(self) -> Tuple[int, np.ndarray]:
        if not self._available:
            raise RuntimeError("RingBuffer exhausted: all buffers in use")
        
        idx = self._available.popleft()
        return idx, self._buffers[idx]

    def release(self, idx: int) -> None:
        if idx in self._available:
            return
        buf = self._buffers[idx]
        if self.use_gpu:
            self._xp.copyto(buf, 0)
        else:
            buf.fill(0)
        self._available.appendleft(idx)

    def __del__(self):
        self._buffers.clear()
        self._available.clear()
        if self.use_gpu and self._xp is not None:
            try:
                self._xp.get_default_memory_pool().free_all_blocks()
            except:
                pass
        gc.collect()


class TileBatchAggregator:
    def __init__(
        self,
        strip_height: int,
        output_width: int,
        num_channels: int = 3,
        use_gpu: bool = False,
        device_id: int = 0
    ):
        self.strip_height = strip_height
        self.output_width = output_width
        self.num_channels = num_channels
        self.use_gpu = use_gpu
        self.device_id = device_id
        
        self._xp = np
        if use_gpu:
            try:
                import cupy as cp
                self._xp = cp
            except ImportError:
                self.use_gpu = False
        
        self._strip_accum = None
        self._strip_weight = None
        self._current_strip_idx = -1
        self._disk_buffer_dir: Optional[Path] = None

    def _allocate_strip(self) -> None:
        shape = (self.strip_height, self.output_width, self.num_channels)
        if self.use_gpu:
            try:
                import cupy as cp
                with cp.cuda.Device(self.device_id):
                    self._strip_accum = cp.zeros(shape, dtype=np.float32)
                    self._strip_weight = cp.zeros((self.strip_height, self.output_width), dtype=np.float32)
                return
            except:
                self.use_gpu = False
                self._xp = np
        
        self._strip_accum = np.zeros(shape, dtype=np.float32)
        self._strip_weight = np.zeros((self.strip_height, self.output_width), dtype=np.float32)

    def add_tile(
        self,
        tile_info: TileInfo,
        sr_tile: np.ndarray,
        gaussian_kernel: np.ndarray,
        sr_effective: int,
        sr_tile_size: int
    ) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
        strip_idx = tile_info.row
        
        if strip_idx != self._current_strip_idx:
            finished = None
            if self._strip_accum is not None:
                finished = (self._current_strip_idx, self._strip_accum, self._strip_weight)
            
            self._current_strip_idx = strip_idx
            self._allocate_strip()
            return finished
        
        out_y = 0
        out_x = tile_info.col * sr_effective
        
        tile_h = min(sr_tile_size, self.strip_height - out_y)
        tile_w = min(sr_tile_size, self.output_width - out_x)
        
        if self.use_gpu:
            import cupy as cp
            if not isinstance(sr_tile, cp.ndarray):
                sr_tile = cp.asarray(sr_tile)
            if not isinstance(gaussian_kernel, cp.ndarray):
                kernel = cp.asarray(gaussian_kernel[:tile_h, :tile_w])
            else:
                kernel = gaussian_kernel[:tile_h, :tile_w]
        else:
            if not isinstance(sr_tile, np.ndarray):
                sr_tile = np.asarray(sr_tile)
            kernel = gaussian_kernel[:tile_h, :tile_w]
        
        kernel_3d = kernel[..., self._xp.newaxis]
        
        tile_data = sr_tile[:tile_h, :tile_w, :].astype(np.float32)
        weighted = tile_data * kernel_3d
        
        self._strip_accum[out_y:out_y + tile_h, out_x:out_x + tile_w, :] += weighted
        self._strip_weight[out_y:out_y + tile_h, out_x:out_x + tile_w] += kernel
        
        del weighted, tile_data, kernel, kernel_3d
        
        return None

    def flush(self) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
        if self._strip_accum is None:
            return None
        
        result = (self._current_strip_idx, self._strip_accum, self._strip_weight)
        
        self._strip_accum = None
        self._strip_weight = None
        self._current_strip_idx = -1
        
        return result


class MemorySafeGaussianStitcher:
    def __init__(
        self,
        output_size: Tuple[int, int],
        tile_size: int = 512,
        overlap: int = 32,
        scale_factor: int = 4,
        blending_sigma: float = 64.0,
        num_channels: int = 3,
        dtype: np.dtype = np.uint8,
        use_gpu: bool = True,
        device_id: int = 0,
        safety_threshold: float = 0.8,
        strip_height_factor: int = 16,
        num_ring_buffers: int = 4,
        enable_disk_spill: bool = True
    ):
        self.output_width, self.output_height = output_size
        self.tile_size = tile_size
        self.overlap = overlap
        self.scale_factor = scale_factor
        self.blending_sigma = blending_sigma
        self.num_channels = num_channels
        self.dtype = dtype
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.safety_threshold = safety_threshold
        self.enable_disk_spill = enable_disk_spill
        
        self.sr_tile_size = tile_size * scale_factor
        self.sr_overlap = overlap * scale_factor
        self.sr_effective = (tile_size - 2 * overlap) * scale_factor
        
        self.total_rows = (self.output_height + self.sr_effective - 1) // self.sr_effective
        self.total_cols = (self.output_width + self.sr_effective - 1) // self.sr_effective
        
        self.strip_height = strip_height_factor * self.sr_effective
        
        self._monitor = GPUMemoryMonitor(device_id, safety_threshold)
        
        self._gaussian_kernel_cpu: Optional[np.ndarray] = None
        self._gaussian_kernel_gpu: Optional[np.ndarray] = None
        
        self._aggregator: Optional[TileBatchAggregator] = None
        self._final_output_cpu: Optional[np.ndarray] = None
        self._final_weight_cpu: Optional[np.ndarray] = None
        
        self._tmp_dir: Optional[Path] = None
        self._strip_files: List[str] = []
        
        self._tiles_processed = 0
        self._total_tiles = 0
        
        self._initialize()

    def _initialize(self) -> None:
        logger.info(f"MemorySafeStitcher init: {self.output_width}x{self.output_height}")
        logger.info(f"Use GPU: {self.use_gpu}, Safety threshold: {self.safety_threshold*100:.0f}%")
        logger.info(f"Strip height: {self.strip_height}px, Total rows: {self.total_rows}")
        
        est_mem = self.estimate_memory_usage(
            self.output_width, self.output_height, self.num_channels,
            self.use_gpu, self.strip_height
        )
        logger.info(f"Estimated peak GPU memory: {est_mem['peak_gpu_mb']:.1f} MB")
        logger.info(f"Estimated CPU memory: {est_mem['cpu_mb']:.1f} MB")
        
        self._gaussian_kernel_cpu = self._create_gaussian_kernel_cpu()
        
        if self.use_gpu:
            try:
                import cupy as cp
                with cp.cuda.Device(self.device_id):
                    self._gaussian_kernel_gpu = cp.asarray(self._gaussian_kernel_cpu)
                logger.info("Gaussian kernel uploaded to GPU")
            except Exception as e:
                logger.warning(f"GPU kernel upload failed: {e}, using CPU mode")
                self.use_gpu = False
        
        self._aggregator = TileBatchAggregator(
            strip_height=self.strip_height,
            output_width=self.output_width,
            num_channels=self.num_channels,
            use_gpu=self.use_gpu,
            device_id=self.device_id
        )
        
        self._final_output_cpu = np.zeros(
            (self.output_height, self.output_width, self.num_channels),
            dtype=np.float32
        )
        self._final_weight_cpu = np.zeros(
            (self.output_height, self.output_width),
            dtype=np.float32
        )
        
        if self.enable_disk_spill:
            self._tmp_dir = Path(tempfile.mkdtemp(prefix="stitcher_"))
            logger.info(f"Disk spill enabled, temp dir: {self._tmp_dir}")
        
        logger.info("MemorySafeGaussianStitcher initialized")

    def _create_gaussian_kernel_cpu(self) -> np.ndarray:
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
        
        return kernel.astype(np.float32)

    def _check_gpu_safety(self) -> bool:
        if not self.use_gpu:
            return True
        
        mem = self._monitor.get_memory_usage()
        if mem["total_mb"] > 0:
            ratio = mem["reserved_mb"] / mem["total_mb"]
            if ratio > self.safety_threshold:
                logger.warning(
                    f"GPU memory usage {ratio*100:.1f}% exceeds threshold "
                    f"{self.safety_threshold*100:.0f}%, forcing GC"
                )
                self._monitor.force_gc()
                
                mem = self._monitor.get_memory_usage()
                ratio = mem["reserved_mb"] / mem["total_mb"]
                if ratio > self.safety_threshold:
                    logger.warning(f"GPU still unsafe after GC, switching strip to CPU")
                    return False
        return True

    def add_tile(self, tile_info: TileInfo, sr_tile: np.ndarray) -> None:
        if self._aggregator is None:
            raise RuntimeError("Stitcher not initialized")
        
        self._tiles_processed += 1
        
        gpu_safe = self._check_gpu_safety()
        use_gpu_for_tile = self.use_gpu and gpu_safe
        
        kernel = self._gaussian_kernel_gpu if use_gpu_for_tile else self._gaussian_kernel_cpu
        if self._aggregator.use_gpu != use_gpu_for_tile:
            self._aggregator.use_gpu = use_gpu_for_tile
        
        try:
            finished_strip = self._aggregator.add_tile(
                tile_info, sr_tile, kernel,
                self.sr_effective, self.sr_tile_size
            )
            
            if finished_strip is not None:
                self._commit_strip(finished_strip)
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "OOM" in str(e):
                logger.error(f"GPU OOM during tile processing: {e}")
                logger.info("Falling back to CPU mode for remaining tiles")
                self.use_gpu = False
                self._aggregator.use_gpu = False
                
                finished_strip = self._aggregator.flush()
                if finished_strip:
                    self._commit_strip(finished_strip)
                
                kernel = self._gaussian_kernel_cpu
                finished_strip = self._aggregator.add_tile(
                    tile_info, sr_tile, kernel,
                    self.sr_effective, self.sr_tile_size
                )
                if finished_strip:
                    self._commit_strip(finished_strip)
            else:
                raise

    def add_tiles_batch(self, tiles: List[Tuple[TileInfo, np.ndarray]]) -> None:
        tiles_sorted = sorted(tiles, key=lambda t: (t[0].row, t[0].col))
        for tile_info, sr_tile in tiles_sorted:
            self.add_tile(tile_info, sr_tile)

    def _commit_strip(self, strip_data: Tuple[int, np.ndarray, np.ndarray]) -> None:
        strip_idx, accum, weight = strip_data
        
        if self.use_gpu:
            try:
                import cupy as cp
                if isinstance(accum, cp.ndarray):
                    accum_cpu = cp.asnumpy(accum)
                    weight_cpu = cp.asnumpy(weight)
                else:
                    accum_cpu = np.asarray(accum)
                    weight_cpu = np.asarray(weight)
            except:
                accum_cpu = np.asarray(accum)
                weight_cpu = np.asarray(weight)
        else:
            accum_cpu = np.asarray(accum)
            weight_cpu = np.asarray(weight)
        
        start_y = strip_idx * self.strip_height
        strip_h = min(self.strip_height, self.output_height - start_y)
        
        accum_cpu = accum_cpu[:strip_h, :, :]
        weight_cpu = weight_cpu[:strip_h, :]
        
        end_y = start_y + strip_h
        self._final_output_cpu[start_y:end_y, :, :] += accum_cpu
        self._final_weight_cpu[start_y:end_y, :] += weight_cpu
        
        if self._tiles_processed % 100 == 0:
            logger.info(f"Processed {self._tiles_processed} tiles, committed strip {strip_idx}")
            mem = self._monitor.get_memory_usage()
            if mem["reserved_mb"] > 0:
                logger.debug(f"GPU memory: {mem['reserved_mb']:.1f} MB / {mem['total_mb']:.1f} MB")
        
        del accum, weight, accum_cpu, weight_cpu
        
        if self.use_gpu:
            self._monitor.force_gc()

    def finalize(self) -> np.ndarray:
        logger.info("Finalizing memory-safe stitching...")
        
        last_strip = self._aggregator.flush()
        if last_strip:
            self._commit_strip(last_strip)
        
        self._monitor.force_gc()
        
        logger.info("Performing final normalization on CPU...")
        weight_3d = self._final_weight_cpu[..., np.newaxis]
        weight_3d = np.maximum(weight_3d, 1e-8)
        
        result = self._final_output_cpu / weight_3d
        result = np.clip(result, 0, 255)
        result = result.astype(self.dtype)
        
        logger.info(f"Stitching complete, output shape: {result.shape}")
        
        self._cleanup()
        
        return result

    def _cleanup(self) -> None:
        if self._tmp_dir and self._tmp_dir.exists():
            try:
                import shutil
                shutil.rmtree(self._tmp_dir)
            except:
                pass
        
        self._gaussian_kernel_gpu = None
        self._aggregator = None
        
        self._monitor.force_gc()
        
        gc.collect()
        
        logger.info("Stitcher resources cleaned up")

    def get_progress(self) -> dict:
        return {
            "tiles_processed": self._tiles_processed,
            "total_tiles_est": self.total_rows * self.total_cols,
            "progress_pct": (self._tiles_processed / max(1, self.total_rows * self.total_cols)) * 100
        }

    def __del__(self):
        try:
            self._cleanup()
        except:
            pass

    @staticmethod
    def estimate_memory_usage(
        width: int,
        height: int,
        num_channels: int = 3,
        use_gpu: bool = False,
        strip_height: int = 4096
    ) -> dict:
        strip_pixels = strip_height * width
        output_pixels = width * height
        
        strip_gpu_mem = strip_pixels * num_channels * 4 + strip_pixels * 4
        kernel_mem = (512 * 4) ** 2 * 4
        ring_buffer_mem = 4 * strip_gpu_mem
        
        cpu_mem = output_pixels * num_channels * 4 + output_pixels * 4
        
        peak_gpu = strip_gpu_mem + kernel_mem + ring_buffer_mem if use_gpu else 0
        
        return {
            "strip_gpu_mb": strip_gpu_mem / (1024 * 1024),
            "peak_gpu_mb": peak_gpu / (1024 * 1024),
            "cpu_mb": cpu_mem / (1024 * 1024),
            "total_gb": (peak_gpu + cpu_mem) / (1024 ** 3)
        }
