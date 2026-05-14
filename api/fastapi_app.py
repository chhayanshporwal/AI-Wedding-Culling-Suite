# api/fastapi_app.py
# React-friendly FastAPI wrapper for the wedding-photo culling pipeline
import os
import zipfile
import tempfile
import csv
import json
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from pipeline import run_filtering
from scripts.enroll_vips import enroll
from filters.face_id_filter import load_vip_embeddings
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# ---------------- FastAPI APP ----------------
app = FastAPI(
    title="Wedding Image Selector API",
    version="1.0.0",
    description="Automated culling of wedding photos with VIP face-ID exemption",
)

# Allow React dev server (adjust origin list for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-react-site.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Pydantic models ----------------
class FilterRequest(BaseModel):
    input_folder: str
    workers: Optional[int] = None


class PersonFilterRequest(BaseModel):
    input_folder: str
    persons: List[str]
    workers: Optional[int] = None


# ---------------- UTILS ----------------
def _csv_to_json(csv_path: str) -> List[Dict[str, Any]]:
    """Convert log.csv → JSON array for React table."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


# ---------------- ROUTES ----------------
@app.get("/ping")
def ping() -> Dict[str, str]:
    """Health check."""
    return {"status": "ok"}


@app.post("/upload-profiles")
async def upload_profiles(
    person: str = Form(..., description="VIP name, e.g. 'Bride'"),
    zipfile_upload: UploadFile = File(..., description="ZIP archive of profile images"),
):
    """Enroll VIP faces from a zip uploaded by React."""
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            zip_path = os.path.join(tmp_dir, zipfile_upload.filename or "profiles.zip")
            with open(zip_path, "wb") as out:
                out.write(await zipfile_upload.read())

            if not zipfile.is_zipfile(zip_path):
                raise HTTPException(status_code=400, detail="Invalid ZIP file")

            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(tmp_dir)

            enroll(person, tmp_dir)  # updates index.json + cache
            return {"enrolled": person}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error during VIP enrollment")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/filter")
def filter_images(req: FilterRequest):
    """Run the full pipeline and return the output folder."""
    if not os.path.isdir(req.input_folder):
        raise HTTPException(status_code=400, detail="Input folder not found")
    try:
        out = run_filtering(
            input_folder=req.input_folder,
            output_base=config.OUTPUT_BASE,
            max_workers=req.workers,
        )
    except Exception as e:
        logger.exception("Filtering failed")
        raise HTTPException(status_code=500, detail=f"Filtering error: {e}")
    if out is None:
        raise HTTPException(status_code=500, detail="No images found")
    return {"output_folder": out}


@app.post("/filter-by-person")
def filter_by_person(req: PersonFilterRequest):
    """
    Run pipeline, then return only the filenames that match requested VIP(s).
    React can fetch the full CSV via /download-log afterwards.
    """
    load_vip_embeddings()
    if not os.path.isdir(req.input_folder):
        raise HTTPException(status_code=400, detail="Input folder not found")
    try:
        out = run_filtering(
            input_folder=req.input_folder,
            output_base=config.OUTPUT_BASE,
            max_workers=req.workers,
        )
    except Exception as e:
        logger.exception("Filtering failed")
        raise HTTPException(status_code=500, detail=f"Filtering error: {e}")
    if out is None:
        raise HTTPException(status_code=500, detail="No images found")

    csv_path = os.path.join(out, "log.csv")
    rows = _csv_to_json(csv_path)

    # Filter by requested persons
    filtered_imgs = []
    for row in rows:
        vip_field = row.get("vip_matches", "")
        if not vip_field:
            continue
        names = [part.split(":")[0] for part in vip_field.split("|") if ":" in part]
        if any(p in names for p in req.persons):
            filtered_imgs.append(row["filename"])

    return {"output_folder": out, "filtered_images": filtered_imgs}


@app.get("/download-log")
def download_log(folder: str):
    """Let React download the CSV log for a finished job."""
    csv_path = os.path.join(folder, "log.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Log file not found")
    with open(csv_path, "rb") as f:
        content = f.read()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=results.csv"},
    )


# Optional: list enrolled VIPs (handy for React dropdown)
@app.get("/vips")
def list_vips():
    from filters.face_id_filter import load_vip_embeddings
    from filters.face_id_filter import VIP_EMBS  # populated by load_vip_embeddings
    load_vip_embeddings()
    return {"vips": list(VIP_EMBS.keys())}