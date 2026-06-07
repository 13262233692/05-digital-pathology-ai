import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from loguru import logger
from celery import current_task, group, chord, chain
import tempfile
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .celery_app import app
from ..wsi_processor.wsi_reader import WSIReader
from ..wsi_processor.tile_extractor import TileExtractor, TileInfo
from ..triton_client.inference_client import TritonInferenceClient
from ..triton_client.batch_processor import BatchProcessor
from ..image_stitcher.memory_safe_stitcher import MemorySafeGaussianStitcher
from ..image_stitcher.ome_tiff_writer import OME_TIFFWriter


def get_config():
    import yaml
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@app.task(bind=True, name="process_wsi", max_retries=3)
def process_wsi(
    self,
    wsi_path: str,
    output_dir: str,
    task_id: Optional[str] = None,
    use_tissue_mask: bool = True
) -> Dict:
    config = get_config()
    wsi_config = config["wsi"]
    sr_config = config["srgan"]
    stitch_config = config["stitching"]
    
    if task_id is None:
        task_id = self.request.id
    
    logger.info(f"Starting WSI processing: {wsi_path}")
    logger.info(f"Task ID: {task_id}")
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        with WSIReader(wsi_path, target_mpp=wsi_config["target_mpp"]) as reader:
            extractor = TileExtractor(
                reader,
                tile_size=wsi_config["tile_size"],
                overlap=wsi_config["overlap"],
                level=wsi_config["pyramid_level"],
                use_tissue_mask=use_tissue_mask
            )
            
            tile_grid = extractor.get_tile_grid()
            total_tiles = len(tile_grid)
            
            logger.info(f"Generated {total_tiles} tiles")
            
            output_dims = extractor.get_output_dimensions()
            origin = extractor.get_output_origin()
            
            metadata = {
                "wsi_path": wsi_path,
                "output_dir": output_dir,
                "total_tiles": total_tiles,
                "output_dimensions": output_dims,
                "origin": origin,
                "tile_size": wsi_config["tile_size"],
                "overlap": wsi_config["overlap"],
                "scale_factor": sr_config["scale_factor"],
            }
            
            metadata_path = Path(output_dir) / f"{task_id}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            batch_size = config["triton"]["batch_size"] * 4
            tile_batches = [
                tile_grid[i:i + batch_size]
                for i in range(0, len(tile_grid), batch_size)
            ]
            
            logger.info(f"Split into {len(tile_batches)} batches")
            
            batch_tasks = []
            for batch_idx, batch in enumerate(tile_batches):
                batch_data = []
                for tile_info in batch:
                    tile_data = extractor.extract_tile(tile_info)
                    batch_data.append({
                        "tile_info": {
                            "tile_id": tile_info.tile_id,
                            "x": tile_info.x,
                            "y": tile_info.y,
                            "width": tile_info.width,
                            "height": tile_info.height,
                            "level": tile_info.level,
                            "row": tile_info.row,
                            "col": tile_info.col,
                            "total_rows": tile_info.total_rows,
                            "total_cols": tile_info.total_cols,
                        },
                        "tile_data": tile_data.tobytes(),
                        "tile_shape": tile_data.shape,
                        "tile_dtype": str(tile_data.dtype),
                    })
                
                batch_path = Path(output_dir) / f"{task_id}_batch_{batch_idx:06d}.pkl"
                with open(batch_path, "wb") as f:
                    pickle.dump(batch_data, f)
                
                batch_tasks.append(
                    process_tile_batch.s(
                        str(batch_path),
                        task_id,
                        batch_idx,
                        len(tile_batches)
                    )
                )
            
            callback = stitch_and_save.s(
                output_dir=output_dir,
                task_id=task_id,
                output_dims=output_dims,
                num_batches=len(tile_batches)
            )
            
            job = chord(group(batch_tasks), callback)
            result = job.apply_async()
            
            return {
                "status": "processing",
                "task_id": task_id,
                "total_tiles": total_tiles,
                "num_batches": len(tile_batches),
                "output_dimensions": output_dims,
                "chord_id": result.id
            }
            
    except Exception as e:
        logger.error(f"Error processing WSI: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@app.task(bind=True, name="process_tile_batch", max_retries=3)
def process_tile_batch(
    self,
    batch_path: str,
    task_id: str,
    batch_idx: int,
    total_batches: int
) -> Dict:
    config = get_config()
    triton_config = config["triton"]
    
    logger.info(f"Processing batch {batch_idx + 1}/{total_batches} for task {task_id}")
    
    try:
        with open(batch_path, "rb") as f:
            batch_data = pickle.load(f)
        
        tile_infos = []
        tiles = []
        for item in batch_data:
            ti = item["tile_info"]
            tile_infos.append(TileInfo(
                tile_id=ti["tile_id"],
                wsi_path="",
                x=ti["x"],
                y=ti["y"],
                width=ti["width"],
                height=ti["height"],
                level=ti["level"],
                row=ti["row"],
                col=ti["col"],
                total_rows=ti["total_rows"],
                total_cols=ti["total_cols"],
            ))
            
            tile = np.frombuffer(item["tile_data"], dtype=np.dtype(item["tile_dtype"]))
            tile = tile.reshape(item["tile_shape"])
            tiles.append(tile)
        
        tiles = np.stack(tiles, axis=0)
        
        with TritonInferenceClient(
            server_url=triton_config["server_url"],
            model_name=triton_config["model_name"],
            model_version=triton_config["model_version"],
            timeout=triton_config["timeout"]
        ) as triton_client:
            sr_tiles = triton_client.infer(tiles)
        
        results = []
        for i, tile_info in enumerate(tile_infos):
            sr_tile = sr_tiles[i]
            results.append({
                "tile_info": {
                    "tile_id": tile_info.tile_id,
                    "row": tile_info.row,
                    "col": tile_info.col,
                },
                "sr_tile": sr_tile.tobytes(),
                "sr_shape": sr_tile.shape,
                "sr_dtype": str(sr_tile.dtype),
            })
        
        output_path = Path(batch_path).parent / f"{task_id}_sr_batch_{batch_idx:06d}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        
        os.remove(batch_path)
        
        progress = ((batch_idx + 1) / total_batches) * 100
        self.update_state(
            state="PROGRESS",
            meta={
                "batch_idx": batch_idx,
                "total_batches": total_batches,
                "progress": progress
            }
        )
        
        return {
            "batch_idx": batch_idx,
            "status": "completed",
            "num_tiles": len(tile_infos),
            "output_path": str(output_path)
        }
        
    except Exception as e:
        logger.error(f"Error processing batch {batch_idx}: {e}")
        raise


@app.task(bind=True, name="stitch_and_save", max_retries=2)
def stitch_and_save(
    self,
    batch_results: List[Dict],
    output_dir: str,
    task_id: str,
    output_dims: Tuple[int, int],
    num_batches: int
) -> Dict:
    config = get_config()
    wsi_config = config["wsi"]
    sr_config = config["srgan"]
    stitch_config = config["stitching"]
    
    logger.info(f"Starting stitching for task {task_id}")
    
    try:
        use_gpu = stitch_config.get("use_gpu", True)
        safety_threshold = stitch_config.get("safety_threshold", 0.8)
        strip_height_factor = stitch_config.get("strip_height_factor", 16)
        
        stitcher = MemorySafeGaussianStitcher(
            output_size=output_dims,
            tile_size=wsi_config["tile_size"],
            overlap=wsi_config["overlap"],
            scale_factor=sr_config["scale_factor"],
            blending_sigma=stitch_config["blending_sigma"],
            use_gpu=use_gpu,
            safety_threshold=safety_threshold,
            strip_height_factor=strip_height_factor,
            enable_disk_spill=True,
        )
        
        all_tiles = []
        for batch_result in batch_results:
            if batch_result.get("status") != "completed":
                logger.warning(f"Skipping incomplete batch: {batch_result}")
                continue
            
            with open(batch_result["output_path"], "rb") as f:
                sr_results = pickle.load(f)
            
            for sr_result in sr_results:
                ti = sr_result["tile_info"]
                tile_info = TileInfo(
                    tile_id=ti["tile_id"],
                    wsi_path="",
                    x=0, y=0, width=0, height=0, level=0,
                    row=ti["row"],
                    col=ti["col"],
                    total_rows=0,
                    total_cols=0,
                )
                
                sr_tile = np.frombuffer(
                    sr_result["sr_tile"],
                    dtype=np.dtype(sr_result["sr_dtype"])
                ).reshape(sr_result["sr_shape"])
                
                all_tiles.append((tile_info, sr_tile))
            
            os.remove(batch_result["output_path"])
        
        logger.info(f"Sorting {len(all_tiles)} tiles by row for memory-efficient streaming")
        all_tiles.sort(key=lambda t: (t[0].row, t[0].col))
        
        for tile_info, sr_tile in all_tiles:
            stitcher.add_tile(tile_info, sr_tile)
        
        del all_tiles
        import gc
        gc.collect()
        
        final_image = stitcher.finalize()
        
        output_path = Path(output_dir) / f"{task_id}_super_resolved.ome.tiff"
        
        writer = OME_TIFFWriter(
            output_path=str(output_path),
            image_shape=(final_image.shape[0], final_image.shape[1]),
            num_channels=final_image.shape[2],
            pixel_size=wsi_config["target_mpp"] / sr_config["scale_factor"],
            compression=stitch_config["compression"],
        )
        writer.write(final_image)
        
        logger.info(f"Successfully saved super-resolved WSI to {output_path}")
        
        return {
            "status": "completed",
            "task_id": task_id,
            "output_path": str(output_path),
            "output_shape": final_image.shape,
            "file_size_mb": output_path.stat().st_size / (1024 * 1024)
        }
        
    except Exception as e:
        logger.error(f"Error during stitching: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise


@app.task(bind=True, name="screen_sr_image", max_retries=2)
def screen_sr_image(
    self,
    image_path: str,
    wsi_name: str,
    task_id: str,
    store_to_milvus: bool = True,
) -> Dict:
    config = get_config()
    screening_config = config.get("screening", {})
    
    logger.info(f"Starting screening for {wsi_name}, task {task_id}")
    
    try:
        import cv2
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        from ..screening.pipeline import ScreeningPipeline
        from ..screening.models.mask_rcnn import NucleusDetector
        from ..screening.morphology.analyzer import MorphologyAnalyzer
        from ..screening.milvus_client.milvus_store import MilvusFeatureStore
        
        detector = NucleusDetector(
            num_classes=screening_config.get("num_classes", 2),
            min_score=screening_config.get("min_score", 0.5),
            mask_threshold=screening_config.get("mask_threshold", 0.5),
            device=screening_config.get("device", "auto"),
        )
        
        weight_path = screening_config.get("model_weights")
        if weight_path and Path(weight_path).exists():
            detector.load_weights(weight_path)
        
        analyzer = MorphologyAnalyzer(
            circularity_threshold=screening_config.get("circularity_threshold", 0.55),
            aspect_ratio_threshold=screening_config.get("aspect_ratio_threshold", 3.0),
            edge_roughness_threshold=screening_config.get("edge_roughness_threshold", 0.4),
            min_nucleus_area=screening_config.get("min_nucleus_area", 100.0),
            max_nucleus_area=screening_config.get("max_nucleus_area", 50000.0),
            abnormality_score_threshold=screening_config.get("abnormality_score_threshold", 0.6),
        )
        
        milvus_store = None
        if store_to_milvus:
            milvus_config = config.get("milvus", {})
            if milvus_config.get("enabled", False):
                milvus_store = MilvusFeatureStore(
                    host=milvus_config.get("host", "localhost"),
                    port=milvus_config.get("port", 19530),
                )
                milvus_store.connect()
                milvus_store._ensure_collection()
        
        pipeline = ScreeningPipeline(
            detector=detector,
            analyzer=analyzer,
            milvus_store=milvus_store,
            tile_size=screening_config.get("tile_size", 2048),
            tile_overlap=screening_config.get("tile_overlap", 128),
        )
        
        result = pipeline.process_large_image(
            image=image,
            wsi_name=wsi_name,
            task_id=task_id,
            store_to_milvus=store_to_milvus,
        )
        
        if milvus_store:
            milvus_store.disconnect()
        
        abnormal_features = [f for f in result["features"] if f.is_abnormal]
        abnormal_summary = []
        for f in abnormal_features:
            abnormal_summary.append({
                "instance_id": f.instance_id,
                "centroid": f.centroid,
                "bbox": f.bbox,
                "circularity": float(f.circularity),
                "aspect_ratio": float(f.aspect_ratio),
                "edge_roughness": float(f.edge_roughness),
                "abnormality_score": float(f.abnormality_score),
                "abnormality_reasons": f.abnormality_reasons,
            })
        
        logger.info(
            f"Screening complete: {result['total_nuclei']} nuclei, "
            f"{result['total_abnormal']} abnormal"
        )
        
        return {
            "status": "completed",
            "task_id": task_id,
            "total_nuclei": result["total_nuclei"],
            "total_abnormal": result["total_abnormal"],
            "abnormal_ratio": result["abnormal_ratio"],
            "abnormal_nuclei": abnormal_summary,
            "stats": result["stats"],
        }
        
    except Exception as e:
        logger.error(f"Error during screening: {e}")
        self.update_state(state="FAILURE", meta={"error": str(e)})
        raise
