import os
import sys
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
import shutil
import uuid
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ..celery_tasks.tasks import process_wsi
from ..celery_tasks.celery_app import app as celery_app


def load_config():
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


config = load_config()
api_config = config["api"]

app = FastAPI(
    title="Pathology SR API",
    description="Digital Pathology Super-Resolution Pipeline API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(api_config["upload_dir"]).mkdir(parents=True, exist_ok=True)
Path(api_config["output_dir"]).mkdir(parents=True, exist_ok=True)


class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class WSIInfoResponse(BaseModel):
    filename: str
    dimensions: List[int]
    level_count: int
    mpp: Optional[List[float]] = None


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "pathology-sr-api"}


@app.get("/api/v1/tasks", response_model=List[dict])
async def list_tasks(limit: int = 50):
    try:
        inspector = celery_app.control.inspect()
        active = inspector.active() or {}
        scheduled = inspector.scheduled() or {}
        reserved = inspector.reserved() or {}
        
        tasks = []
        for worker_name, worker_tasks in active.items():
            for task in worker_tasks:
                tasks.append({
                    "id": task["id"],
                    "name": task["name"],
                    "status": "active",
                    "worker": worker_name,
                    "time_started": task.get("time_started")
                })
        
        for worker_name, worker_tasks in scheduled.items():
            for task in worker_tasks:
                tasks.append({
                    "id": task["request"]["id"],
                    "name": task["request"]["name"],
                    "status": "scheduled",
                    "worker": worker_name,
                })
        
        return tasks[:limit]
    except Exception as e:
        logger.error(f"Error listing tasks: {e}")
        return []


@app.post("/api/v1/wsi/upload", response_model=TaskResponse)
async def upload_wsi(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    use_tissue_mask: bool = True
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    ext = Path(file.filename).suffix.lower()
    supported_ext = ['.svs', '.tiff', '.tif']
    if ext not in supported_ext:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {', '.join(supported_ext)}"
        )
    
    task_id = str(uuid.uuid4())
    
    upload_path = Path(api_config["upload_dir"]) / task_id / file.filename
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size = upload_path.stat().st_size
        if file_size > api_config["max_upload_size"]:
            shutil.rmtree(upload_path.parent)
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {api_config['max_upload_size'] / (1024**3):.1f} GB"
            )
        
        output_dir = Path(api_config["output_dir"]) / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = process_wsi.delay(
            wsi_path=str(upload_path),
            output_dir=str(output_dir),
            task_id=task_id,
            use_tissue_mask=use_tissue_mask
        )
        
        logger.info(f"Submitted task {task_id} for WSI: {file.filename}")
        
        return TaskResponse(
            task_id=task_id,
            status="submitted",
            message="WSI processing task submitted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading WSI: {e}")
        if upload_path.parent.exists():
            shutil.rmtree(upload_path.parent)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/wsi/process/{task_id}", response_model=TaskResponse)
async def process_existing_wsi(
    task_id: str,
    wsi_path: str,
    use_tissue_mask: bool = True
):
    wsi_file = Path(wsi_path)
    if not wsi_file.exists():
        raise HTTPException(status_code=404, detail=f"WSI file not found: {wsi_path}")
    
    output_dir = Path(api_config["output_dir"]) / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = process_wsi.delay(
            wsi_path=str(wsi_file),
            output_dir=str(output_dir),
            task_id=task_id,
            use_tissue_mask=use_tissue_mask
        )
        
        return TaskResponse(
            task_id=task_id,
            status="submitted",
            message="WSI processing task submitted successfully"
        )
    except Exception as e:
        logger.error(f"Error processing WSI: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    try:
        result = process_wsi.AsyncResult(task_id)
        
        status = result.state
        response = TaskStatusResponse(
            task_id=task_id,
            status=status.lower()
        )
        
        if result.ready():
            if result.successful():
                response.result = result.result
            else:
                response.error = str(result.result)
        
        if status == "PROGRESS":
            response.result = result.info
        
        return response
        
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tasks/{task_id}/result/download")
async def download_result(task_id: str):
    output_dir = Path(api_config["output_dir"]) / task_id
    result_files = list(output_dir.glob("*_super_resolved.ome.tiff"))
    
    if not result_files:
        raise HTTPException(status_code=404, detail="Result not found or not ready")
    
    result_file = result_files[0]
    return FileResponse(
        path=str(result_file),
        filename=result_file.name,
        media_type="image/tiff"
    )


@app.get("/api/v1/tasks/{task_id}/metadata")
async def get_task_metadata(task_id: str):
    output_dir = Path(api_config["output_dir"]) / task_id
    metadata_files = list(output_dir.glob("*_metadata.json"))
    
    if not metadata_files:
        raise HTTPException(status_code=404, detail="Metadata not found")
    
    try:
        with open(metadata_files[0], "r") as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/tasks/{task_id}")
async def cancel_task(task_id: str):
    try:
        result = process_wsi.AsyncResult(task_id)
        if not result.ready():
            result.revoke(terminate=True)
        
        task_dir = Path(api_config["output_dir"]) / task_id
        if task_dir.exists():
            shutil.rmtree(task_dir)
        
        upload_dir = Path(api_config["upload_dir"]) / task_id
        if upload_dir.exists():
            shutil.rmtree(upload_dir)
        
        return {"status": "cancelled", "task_id": task_id}
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/system/info")
async def get_system_info():
    try:
        inspector = celery_app.control.inspect()
        stats = inspector.stats() or {}
        
        workers = []
        for worker_name, worker_stats in stats.items():
            workers.append({
                "name": worker_name,
                "concurrency": worker_stats.get("pool", {}).get("max-concurrency", 0),
                "processes": worker_stats.get("pool", {}).get("processes", []),
                "completed_tasks": worker_stats.get("total", 0),
            })
        
        return {
            "workers": workers,
            "upload_dir": api_config["upload_dir"],
            "output_dir": api_config["output_dir"],
            "supported_formats": config["wsi"]["supported_formats"],
            "tile_size": config["wsi"]["tile_size"],
            "overlap": config["wsi"]["overlap"],
            "scale_factor": config["srgan"]["scale_factor"],
        }
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=api_config["host"],
        port=api_config["port"]
    )
