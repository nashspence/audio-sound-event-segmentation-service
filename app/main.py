from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.service import ATSTSEDService


service = ATSTSEDService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.load()
    yield


app = FastAPI(
    title="ATST-SED Service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "atst-sed-service",
        "health": "/healthz",
        "detect": "/v1/detect",
    }


@app.get("/healthz")
def healthz():
    payload = service.health()
    status_code = 200 if payload["ready"] else 503
    return JSONResponse(content=payload, status_code=status_code)


@app.post("/v1/detect")
async def detect(
    file: UploadFile = File(...),
    include_speech: bool = Query(default=False),
):
    if not service.ready:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    filename = file.filename or "upload"
    suffix = Path(filename).suffix or ".bin"

    with tempfile.TemporaryDirectory(prefix="atst-sed-upload-") as tmpdir:
        upload_path = Path(tmpdir) / f"upload{suffix}"
        with upload_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        return service.detect(upload_path, filename, include_speech=include_speech)

