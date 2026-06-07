import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger
from pathlib import Path
import time
import uuid

from .models.mask_rcnn import NucleusDetector
from .morphology.analyzer import MorphologyAnalyzer, NucleusFeatures
from .milvus_client.milvus_store import MilvusFeatureStore, NucleusRecord


class ScreeningPipeline:
    def __init__(
        self,
        detector: NucleusDetector,
        analyzer: MorphologyAnalyzer,
        milvus_store: Optional[MilvusFeatureStore] = None,
        tile_size: int = 2048,
        tile_overlap: int = 128,
        min_nucleus_area: float = 100.0,
        nms_iou_threshold: float = 0.5,
    ):
        self.detector = detector
        self.analyzer = analyzer
        self.milvus_store = milvus_store
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.min_nucleus_area = min_nucleus_area
        self.nms_iou_threshold = nms_iou_threshold
        
        self._stats = {
            "tiles_processed": 0,
            "nuclei_detected": 0,
            "nuclei_analyzed": 0,
            "abnormal_nuclei": 0,
            "records_stored": 0,
        }

    def _split_into_tiles(
        self, image: np.ndarray
    ) -> List[Tuple[np.ndarray, int, int]]:
        h, w = image.shape[:2]
        step = self.tile_size - self.tile_overlap
        
        tiles = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                y_end = min(y + self.tile_size, h)
                x_end = min(x + self.tile_size, w)
                
                tile = image[y:y_end, x:x_end]
                
                if tile.shape[0] < 64 or tile.shape[1] < 64:
                    continue
                
                pad_h = self.tile_size - tile.shape[0]
                pad_w = self.tile_size - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = np.pad(
                        tile,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode="reflect",
                    )
                
                tiles.append((tile[:self.tile_size, :self.tile_size], y, x))
        
        return tiles

    def _merge_tile_results(
        self,
        tile_results: List[Dict],
        offsets: List[Tuple[int, int]],
    ) -> Dict:
        all_boxes = []
        all_masks = []
        all_scores = []
        all_labels = []
        
        for result, (off_y, off_x) in zip(tile_results, offsets):
            n = result["num_instances"]
            if n == 0:
                continue
            
            boxes = result["boxes"].copy()
            boxes[:, 0] += off_x
            boxes[:, 1] += off_y
            boxes[:, 2] += off_x
            boxes[:, 3] += off_y
            
            padded_masks = np.zeros(
                (n, result["masks"].shape[1] + off_y, result["masks"].shape[2] + off_x),
                dtype=np.uint8,
            )
            padded_masks[:, off_y:, off_x:] = result["masks"]
            
            all_boxes.append(boxes)
            all_masks.append(padded_masks)
            all_scores.append(result["scores"])
            all_labels.append(result["labels"])
        
        if not all_boxes:
            return {
                "boxes": np.empty((0, 4)),
                "masks": np.empty((0, 0, 0), dtype=np.uint8),
                "scores": np.empty(0),
                "labels": np.empty(0),
                "num_instances": 0,
            }
        
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        keep = self._nms(boxes, scores, self.nms_iou_threshold)
        
        return {
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
            "num_instances": len(keep),
        }

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, threshold: float
    ) -> np.ndarray:
        if len(boxes) == 0:
            return np.array([], dtype=int)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=int)

    def process_tile(
        self,
        image: np.ndarray,
        wsi_name: str = "",
        task_id: str = "",
        tile_row: int = 0,
        tile_col: int = 0,
        image_offset: Tuple[int, int] = (0, 0),
        store_to_milvus: bool = True,
    ) -> Dict:
        start_time = time.time()
        
        detection_result = self.detector.predict(image)
        self._stats["tiles_processed"] += 1
        self._stats["nuclei_detected"] += detection_result["num_instances"]
        
        if detection_result["num_instances"] == 0:
            return {
                "num_nuclei": 0,
                "num_abnormal": 0,
                "features": [],
                "processing_time_ms": (time.time() - start_time) * 1000,
            }
        
        contours = self.detector.extract_contours(detection_result["masks"])
        
        nuclei_features = self.analyzer.analyze_instances(
            masks=detection_result["masks"],
            contours=contours,
            scores=detection_result["scores"],
            image_offset=image_offset,
        )
        
        self._stats["nuclei_analyzed"] += len(nuclei_features)
        
        abnormal_features = [f for f in nuclei_features if f.is_abnormal]
        self._stats["abnormal_nuclei"] += len(abnormal_features)
        
        if store_to_milvus and self.milvus_store and abnormal_features:
            records = self._features_to_records(
                abnormal_features, wsi_name, task_id, tile_row, tile_col
            )
            try:
                inserted = self.milvus_store.insert_records(records)
                self._stats["records_stored"] += inserted
            except Exception as e:
                logger.error(f"Failed to store records in Milvus: {e}")
        
        elapsed = (time.time() - start_time) * 1000
        
        return {
            "num_nuclei": len(nuclei_features),
            "num_abnormal": len(abnormal_features),
            "features": nuclei_features,
            "processing_time_ms": elapsed,
        }

    def process_large_image(
        self,
        image: np.ndarray,
        wsi_name: str = "",
        task_id: str = "",
        store_to_milvus: bool = True,
    ) -> Dict:
        logger.info(f"Processing large image: {image.shape}, tile_size={self.tile_size}")
        
        tiles = self._split_into_tiles(image)
        logger.info(f"Split into {len(tiles)} tiles")
        
        all_features = []
        total_abnormal = 0
        
        for idx, (tile, off_y, off_x) in enumerate(tiles):
            result = self.process_tile(
                image=tile,
                wsi_name=wsi_name,
                task_id=task_id,
                tile_row=off_y // self.tile_size,
                tile_col=off_x // self.tile_size,
                image_offset=(off_y, off_x),
                store_to_milvus=store_to_milvus,
            )
            
            all_features.extend(result["features"])
            total_abnormal += result["num_abnormal"]
            
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"Processed {idx + 1}/{len(tiles)} tiles, "
                    f"{len(all_features)} nuclei, {total_abnormal} abnormal"
                )
        
        if self.milvus_store:
            self.milvus_store.flush()
        
        return {
            "total_nuclei": len(all_features),
            "total_abnormal": total_abnormal,
            "abnormal_ratio": total_abnormal / max(1, len(all_features)),
            "features": all_features,
            "stats": self._stats.copy(),
        }

    def _features_to_records(
        self,
        features: List[NucleusFeatures],
        wsi_name: str,
        task_id: str,
        tile_row: int,
        tile_col: int,
    ) -> List[NucleusRecord]:
        records = []
        for f in features:
            polygon_str = json.dumps(
                self.analyzer.polygon_to_tuple(f.contour)
            )
            
            record = NucleusRecord(
                record_id=f"{task_id}_{f.instance_id}_{uuid.uuid4().hex[:8]}",
                wsi_name=wsi_name,
                task_id=task_id,
                tile_row=tile_row,
                tile_col=tile_col,
                centroid_x=f.centroid[0],
                centroid_y=f.centroid[1],
                bbox_x=f.bbox[0],
                bbox_y=f.bbox[1],
                bbox_w=f.bbox[2],
                bbox_h=f.bbox[3],
                area=f.area,
                perimeter=f.perimeter,
                circularity=f.circularity,
                aspect_ratio=f.aspect_ratio,
                edge_roughness=f.edge_roughness,
                solidity=f.solidity,
                convexity=f.convexity,
                eccentricity=f.eccentricity,
                abnormality_score=f.abnormality_score,
                abnormality_reasons="; ".join(f.abnormality_reasons),
                feature_vector=f.feature_vector,
                polygon_coords=polygon_str,
            )
            records.append(record)
        
        return records

    def get_stats(self) -> Dict:
        return self._stats.copy()

    def reset_stats(self) -> None:
        for key in self._stats:
            self._stats[key] = 0

    @classmethod
    def from_config(cls, config: dict) -> "ScreeningPipeline":
        screening_config = config.get("screening", {})
        
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
        
        milvus_config = config.get("milvus", {})
        milvus_store = None
        if milvus_config.get("enabled", False):
            milvus_store = MilvusFeatureStore(
                host=milvus_config.get("host", "localhost"),
                port=milvus_config.get("port", 19530),
                collection_name=milvus_config.get("collection_name", MILVUS_COLLECTION_NAME),
            )
        
        return cls(
            detector=detector,
            analyzer=analyzer,
            milvus_store=milvus_store,
            tile_size=screening_config.get("tile_size", 2048),
            tile_overlap=screening_config.get("tile_overlap", 128),
            nms_iou_threshold=screening_config.get("nms_iou_threshold", 0.5),
        )
