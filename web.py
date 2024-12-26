from fastapi import FastAPI, BackgroundTasks, Query, HTTPException, APIRouter
import subprocess
import uvicorn
import os
from typing import Dict
from datetime import datetime
import sys
nerf_path = os.path.join(os.getcwd(), 'ER-NeRF')
print(nerf_path)
sys.path.append(nerf_path)

app = FastAPI()
router = APIRouter(prefix="/nerf")

training_processes: Dict[str, subprocess.Popen] = {}

def log_infer_status(status, log_file_path):
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp}|!{status}")

def get_log_file_path(log_directory, file_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_directory, f"{file_name}_{timestamp}.txt")

def run_infer_script(command, log_file_path, results_log_path):
    try:
        with open(log_file_path, "a") as log_file:
            process = subprocess.Popen(command, stdout=log_file, stderr=log_file, text=True, bufsize=1)
            log_infer_status("running|!", results_log_path)
            process.wait()
            if process.returncode == 0:
                log_infer_status(f"success\n", results_log_path)
            else:
                log_infer_status(f"fail\n", results_log_path)
    except Exception as e:
        log_infer_status(f"fail\n", results_log_path)
        with open(log_file_path, "a") as log_file:
            log_file.write(f"Error occurred: {str(e)}\n")

@router.get("/process")
async def process_data(
    train_name: str = Query(..., description="The name of the training data."),
    task: int = Query(1, description="Task ID to execute, default is 1."),
    background_tasks: BackgroundTasks = None
):
    if not train_name:
        raise HTTPException(status_code=400, detail="Parameter 'train_name' is required.")
    if task < 1:
        raise HTTPException(status_code=400, detail="Parameter 'task' must be a positive integer.")
    
    log_file_path = get_log_file_path(f"process_{train_name}")
    command = f"python data_utils/process.py data/{train_name}/{train_name}.mp4 --log_file {log_file_path} --task {task}"

    background_tasks.add_task(run_script, command, "process.py", log_file_path)
    
    return {
        "message": f"Processing started in background for train_name: {train_name}, task: {task}",
        "log_file": log_file_path
    }

@router.get("/train")
async def train(train_name: str = Query(...), background_tasks: BackgroundTasks = None):
    if train_name in training_processes:
        raise HTTPException(status_code=400, detail=f"Training for '{train_name}' is already running.")
    
    log_file_path = get_log_file_path(f"train_{train_name}")
    command = ["python", "./scripts/train.py", train_name, log_file_path]

    background_tasks.add_task(run_script, command, "train.py", log_file_path, train_name)
    return {"message": f"Training started in background for train_name: {train_name}", "log_file": log_file_path}

@router.get("/infer")
async def infer(
    digitalHumanName: str = Query(..., description="Name of the digital human model."),
    testAudioName: str = Query(..., description="Name of the test audio file."),
    inference_part: str = Query(..., description="Part for inference (e.g., 'head')."),
    background_tasks: BackgroundTasks = None
):
    infer_id = f"infer_{digitalHumanName}_talk_{testAudioName}"
    log_directory = './_DEBUG/logs/'
    os.makedirs(log_directory, exist_ok=True)
    log_file_path = get_log_file_path(log_directory, infer_id)
    command = [
        "python", "./ER-NeRF/scripts/infer.py",
        "--digitalHumanName", digitalHumanName,
        "--testAudioName", testAudioName,
        "--inference_part", inference_part,
        "--log_file", log_file_path
    ]
    results_log_path = "./_DEBUG/res/results.txt"
    background_tasks.add_task(run_infer_script, command, log_file_path, results_log_path)
    return {
        "message": f"Inference started in background with digitalHumanName: {digitalHumanName}, testAudioName: {testAudioName}, inference_part: {inference_part}",
        "log_file": log_file_path
    }


@router.get("/stop_train")
async def stop_train(train_name: str = Query(...)):
    process = training_processes.get(train_name)
    
    if not process:
        raise HTTPException(status_code=404, detail=f"No active training process found for '{train_name}'.")
    
    process.terminate()
    process.wait()
    del training_processes[train_name]
    
    return {"message": f"Training for '{train_name}' has been stopped."}

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("web:app", host="127.0.0.1", port=8000, reload=False)
