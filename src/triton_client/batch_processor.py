import numpy as np
from typing import List, Tuple, Optional, Callable
from collections import deque
from loguru import logger
import concurrent.futures
import threading
import time

from .inference_client import TritonInferenceClient
from ..wsi_processor.tile_extractor import TileInfo


class BatchProcessor:
    def __init__(
        self,
        triton_client: TritonInferenceClient,
        batch_size: int = 8,
        max_workers: int = 4,
        max_queue_size: int = 128
    ):
        self.triton_client = triton_client
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        self._queue: deque = deque()
        self._lock = threading.Lock()
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._results: dict = {}
        self._pending: dict = {}
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def _batch_worker(self):
        while not self._stop_event.is_set():
            batch_items = []
            
            with self._lock:
                while len(self._queue) > 0 and len(batch_items) < self.batch_size:
                    batch_items.append(self._queue.popleft())
            
            if not batch_items:
                time.sleep(0.01)
                continue
            
            tile_infos = [item[0] for item in batch_items]
            tiles = np.stack([item[1] for item in batch_items], axis=0)
            
            try:
                sr_tiles = self.triton_client.infer(tiles)
                
                for i, tile_info in enumerate(tile_infos):
                    self._results[tile_info.tile_id] = sr_tiles[i]
                    
                    if tile_info.tile_id in self._pending:
                        callback, user_data = self._pending.pop(tile_info.tile_id)
                        if callback:
                            callback(tile_info, sr_tiles[i], user_data)
                            
            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                for tile_info in tile_infos:
                    if tile_info.tile_id in self._pending:
                        callback, user_data = self._pending.pop(tile_info.tile_id)
                        if callback:
                            callback(tile_info, None, user_data, error=e)

    def start(self):
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._batch_worker, daemon=True)
            self._worker_thread.start()
            logger.info("Batch processor started")

    def stop(self):
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("Batch processor stopped")

    def submit(
        self,
        tile_info: TileInfo,
        tile_data: np.ndarray,
        callback: Optional[Callable] = None,
        user_data: dict = None
    ) -> str:
        with self._lock:
            if len(self._queue) >= self.max_queue_size:
                logger.warning("Batch queue full, waiting...")
                while len(self._queue) >= self.max_queue_size:
                    time.sleep(0.1)
            
            self._queue.append((tile_info, tile_data))
            self._pending[tile_info.tile_id] = (callback, user_data)
        
        return tile_info.tile_id

    def submit_batch(
        self,
        tile_items: List[Tuple[TileInfo, np.ndarray]],
        callback: Optional[Callable] = None,
        user_data: dict = None
    ) -> List[str]:
        tile_ids = []
        for tile_info, tile_data in tile_items:
            tid = self.submit(tile_info, tile_data, callback, user_data)
            tile_ids.append(tid)
        return tile_ids

    def get_result(self, tile_id: str, timeout: float = 30.0) -> Optional[np.ndarray]:
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if tile_id in self._results:
                    return self._results.pop(tile_id)
            time.sleep(0.01)
        raise TimeoutError(f"Timeout waiting for result {tile_id}")

    def get_results(self, tile_ids: List[str], timeout: float = 60.0) -> dict:
        results = {}
        for tile_id in tile_ids:
            try:
                results[tile_id] = self.get_result(tile_id, timeout)
            except TimeoutError:
                results[tile_id] = None
                logger.warning(f"Timeout for tile {tile_id}")
        return results

    def wait_all(self, timeout: float = 120.0) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self._lock:
                if len(self._queue) == 0 and len(self._pending) == 0:
                    return True
            time.sleep(0.1)
        return False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False
