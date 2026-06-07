import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
import json
import time

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("pymilvus not available. Milvus storage will be disabled.")


MILVUS_COLLECTION_NAME = "nucleus_abnormalities"
FEATURE_DIM = 10


@dataclass
class NucleusRecord:
    record_id: str
    wsi_name: str
    task_id: str
    tile_row: int
    tile_col: int
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    area: float
    perimeter: float
    circularity: float
    aspect_ratio: float
    edge_roughness: float
    solidity: float
    convexity: float
    eccentricity: float
    abnormality_score: float
    abnormality_reasons: str
    feature_vector: np.ndarray
    polygon_coords: str


class MilvusFeatureStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = MILVUS_COLLECTION_NAME,
        feature_dim: int = FEATURE_DIM,
        alias: str = "default",
    ):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.feature_dim = feature_dim
        self.alias = alias
        
        self._connected = False
        self._collection: Optional[Collection] = None

    def connect(self) -> bool:
        if not MILVUS_AVAILABLE:
            logger.error("pymilvus not installed, cannot connect to Milvus")
            return False
        
        try:
            connections.connect(
                alias=self.alias,
                host=self.host,
                port=self.port,
            )
            self._connected = True
            logger.info(f"Connected to Milvus at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            self._connected = False
            return False

    def _ensure_collection(self) -> None:
        if not self._connected:
            raise RuntimeError("Not connected to Milvus")
        
        if utility.has_collection(self.collection_name, using=self.alias):
            self._collection = Collection(self.collection_name, using=self.alias)
            return
        
        pk_field = FieldSchema(
            name="record_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            max_length=128,
        )
        
        wsi_field = FieldSchema(name="wsi_name", dtype=DataType.VARCHAR, max_length=256)
        task_field = FieldSchema(name="task_id", dtype=DataType.VARCHAR, max_length=128)
        
        tile_row_field = FieldSchema(name="tile_row", dtype=DataType.INT32)
        tile_col_field = FieldSchema(name="tile_col", dtype=DataType.INT32)
        
        cx_field = FieldSchema(name="centroid_x", dtype=DataType.FLOAT)
        cy_field = FieldSchema(name="centroid_y", dtype=DataType.FLOAT)
        
        bx_field = FieldSchema(name="bbox_x", dtype=DataType.INT32)
        by_field = FieldSchema(name="bbox_y", dtype=DataType.INT32)
        bw_field = FieldSchema(name="bbox_w", dtype=DataType.INT32)
        bh_field = FieldSchema(name="bbox_h", dtype=DataType.INT32)
        
        area_field = FieldSchema(name="area", dtype=DataType.FLOAT)
        perimeter_field = FieldSchema(name="perimeter", dtype=DataType.FLOAT)
        circularity_field = FieldSchema(name="circularity", dtype=DataType.FLOAT)
        aspect_ratio_field = FieldSchema(name="aspect_ratio", dtype=DataType.FLOAT)
        edge_roughness_field = FieldSchema(name="edge_roughness", dtype=DataType.FLOAT)
        solidity_field = FieldSchema(name="solidity", dtype=DataType.FLOAT)
        convexity_field = FieldSchema(name="convexity", dtype=DataType.FLOAT)
        eccentricity_field = FieldSchema(name="eccentricity", dtype=DataType.FLOAT)
        
        score_field = FieldSchema(name="abnormality_score", dtype=DataType.FLOAT)
        reasons_field = FieldSchema(name="abnormality_reasons", dtype=DataType.VARCHAR, max_length=512)
        
        feature_field = FieldSchema(
            name="feature_vector",
            dtype=DataType.FLOAT_VECTOR,
            dim=self.feature_dim,
        )
        
        polygon_field = FieldSchema(name="polygon_coords", dtype=DataType.VARCHAR, max_length=4096)
        
        schema = CollectionSchema(
            fields=[
                pk_field, wsi_field, task_field,
                tile_row_field, tile_col_field,
                cx_field, cy_field,
                bx_field, by_field, bw_field, bh_field,
                area_field, perimeter_field,
                circularity_field, aspect_ratio_field,
                edge_roughness_field, solidity_field,
                convexity_field, eccentricity_field,
                score_field, reasons_field,
                feature_field, polygon_field,
            ],
            description="Nucleus abnormality feature store for pathology screening",
        )
        
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self.alias,
        )
        
        logger.info(f"Created Milvus collection: {self.collection_name}")

    def create_index(self) -> None:
        if self._collection is None:
            self._ensure_collection()
        
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        
        self._collection.create_index(
            field_name="feature_vector",
            index_params=index_params,
        )
        
        score_index = {
            "metric_type": "L2",
            "index_type": "STL_SORT",
        }
        self._collection.create_index(
            field_name="abnormality_score",
            index_params=score_index,
        )
        
        logger.info("Created Milvus indexes")

    def insert_records(
        self,
        records: List[NucleusRecord],
    ) -> int:
        if not self._connected:
            raise RuntimeError("Not connected to Milvus")
        
        if self._collection is None:
            self._ensure_collection()
        
        if not records:
            return 0
        
        data = [
            [r.record_id for r in records],
            [r.wsi_name for r in records],
            [r.task_id for r in records],
            [r.tile_row for r in records],
            [r.tile_col for r in records],
            [r.centroid_x for r in records],
            [r.centroid_y for r in records],
            [r.bbox_x for r in records],
            [r.bbox_y for r in records],
            [r.bbox_w for r in records],
            [r.bbox_h for r in records],
            [r.area for r in records],
            [r.perimeter for r in records],
            [r.circularity for r in records],
            [r.aspect_ratio for r in records],
            [r.edge_roughness for r in records],
            [r.solidity for r in records],
            [r.convexity for r in records],
            [r.eccentricity for r in records],
            [r.abnormality_score for r in records],
            [r.abnormality_reasons for r in records],
            [r.feature_vector.tolist() for r in records],
            [r.polygon_coords for r in records],
        ]
        
        try:
            result = self._collection.insert(data)
            logger.info(f"Inserted {len(records)} records into Milvus")
            return result.insert_count
        except Exception as e:
            logger.error(f"Failed to insert records: {e}")
            raise

    def search_similar(
        self,
        query_vector: np.ndarray,
        top_k: int = 20,
        filter_expr: Optional[str] = None,
    ) -> List[Dict]:
        if self._collection is None:
            self._ensure_collection()
        
        self._collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 64},
        }
        
        results = self._collection.search(
            data=[query_vector.tolist()],
            anns_field="feature_vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=[
                "record_id", "wsi_name", "centroid_x", "centroid_y",
                "circularity", "aspect_ratio", "edge_roughness",
                "abnormality_score", "abnormality_reasons",
            ],
        )
        
        matches = []
        for hit in results[0]:
            matches.append({
                "record_id": hit.entity.get("record_id"),
                "wsi_name": hit.entity.get("wsi_name"),
                "centroid": (hit.entity.get("centroid_x"), hit.entity.get("centroid_y")),
                "circularity": hit.entity.get("circularity"),
                "aspect_ratio": hit.entity.get("aspect_ratio"),
                "edge_roughness": hit.entity.get("edge_roughness"),
                "abnormality_score": hit.entity.get("abnormality_score"),
                "abnormality_reasons": hit.entity.get("abnormality_reasons"),
                "distance": hit.distance,
            })
        
        return matches

    def query_abnormal(
        self,
        wsi_name: Optional[str] = None,
        min_score: float = 0.6,
        limit: int = 100,
    ) -> List[Dict]:
        if self._collection is None:
            self._ensure_collection()
        
        self._collection.load()
        
        expr = f"abnormality_score >= {min_score}"
        if wsi_name:
            expr += f' && wsi_name == "{wsi_name}"'
        
        results = self._collection.query(
            expr=expr,
            output_fields=[
                "record_id", "wsi_name", "task_id",
                "centroid_x", "centroid_y",
                "bbox_x", "bbox_y", "bbox_w", "bbox_h",
                "circularity", "aspect_ratio", "edge_roughness",
                "abnormality_score", "abnormality_reasons",
                "polygon_coords",
            ],
            limit=limit,
        )
        
        return results

    def flush(self) -> None:
        if self._collection:
            self._collection.flush()

    def disconnect(self) -> None:
        if self._connected:
            connections.disconnect(self.alias)
            self._connected = False
            logger.info("Disconnected from Milvus")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
