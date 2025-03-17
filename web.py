import uvicorn
from datetime import datetime
from fastapi import FastAPI, APIRouter, Query, BackgroundTasks
from scripts.inference import submit_task, init_system
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("✅ 应用启动...")
    init_system()
    yield
    print("🛑 应用关闭...")

app = FastAPI(lifespan=lifespan)
router = APIRouter(prefix="/nerf")

@router.get("/infer")
async def infer(
    digitalHumanName: str = Query(..., description="Name of the digital human model."),
    testAudioName: str = Query(..., description="Name of the test audio file."),
    inferencePart: str = Query(..., description="Part for inference (e.g., 'head')."),
    publicId: str = Query(None, description="optional"),
    background_tasks: BackgroundTasks = None
):
    background_tasks.add_task(submit_task, digitalHumanName, testAudioName, inferencePart, publicId)
    return {
        "message": f"Inference started in background with digitalHumanName: {digitalHumanName}, \
            testAudioName: {testAudioName}, inferencePart: {inferencePart}, publicId: {publicId}",
    }

@router.get("/test")
async def test():
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {"message": f"Tested at {timestamp}"}

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("web:app", host="127.0.0.1", port=8000, reload=False)
