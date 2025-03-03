from fastapi import FastAPI, BackgroundTasks, Query, HTTPException, APIRouter, Request
import subprocess
import uvicorn
import os, sys
from typing import Dict
from datetime import datetime
from multiprocessing import shared_memory
import struct
from starlette.responses import JSONResponse
from scripts.inference import run_infer, terminate_infer, get_infer_progress

app = FastAPI()
router = APIRouter(prefix="/nerf")

@router.get("/infer")
async def infer(
    digitalHumanName: str = Query(..., description="Name of the digital human model."),
    testAudioName: str = Query(..., description="Name of the test audio file."),
    inference_part: str = Query(..., description="Part for inference (e.g., 'head')."),
    publicId: str = Query(None, description="optional"),
    background_tasks: BackgroundTasks = None,
):
    background_tasks.add_task(run_infer, digitalHumanName, testAudioName, inference_part, publicId)
    return {
        "message": f"Inference started in background with digitalHumanName: {digitalHumanName}, \
            testAudioName: {testAudioName}, inference_part: {inference_part}",
    }

@router.get("/infer_progress")
async def infer_progress(
    digitalHumanName: str = Query(..., description="Name of the digital human model."),
    testAudioName: str = Query(..., description="Name of the test audio file."),
    inference_part: str = Query(..., description="Part for inference (e.g., 'head').")
):
    shared_total, shared_step = get_infer_progress(digitalHumanName, testAudioName, inference_part)
    return {
        "message": f"total: {shared_total}, step: {shared_step}"
    }


@app.get("/terminate_infer")
async def terminate_inference(
    digitalHumanName: str = Query(..., description="Name of the digital human model to terminate.")
):
    result = terminate_infer(digitalHumanName)
    return {"message": result}


@router.get("/test")
async def test():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"message": f"Tested at {timestamp}"}


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("web:app", host="127.0.0.1", port=8000, reload=False)
