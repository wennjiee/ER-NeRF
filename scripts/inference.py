import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print('载入inference.py', os.getcwd())
import gc
import uuid
import torch
import struct
import logging
import subprocess
from typing import Dict
from ffmpy import FFmpeg
from datetime import datetime
from multiprocessing import shared_memory
from data_utils.hubert_processor import HubertProcessor
from main import main
import contextlib
import shutil

import time
import uuid
import threading
import queue
import multiprocessing

MAX_WORKERS = 2
TOTAL_CPU_CORES = 8
tasks = {}  # 统一维护任务状态和进程对象
task_queue = queue.Queue()
task_lock = threading.Lock()
status_update_queue = multiprocessing.Queue()

import time
import threading
import psutil
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

inferring_processes: Dict[str, str] = {}

def setup_logger(id: int, infer_file_path: str) -> logging.Logger:
    logger = logging.getLogger(f"infer_{id}")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(infer_file_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def close_logger(logger: logging.Logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def log_status(res_file_path, status):
    log_dir = os.path.dirname(res_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(res_file_path, "a") as log_file:
        log_file.write(f"{timestamp}|!{status}")

def terminate_infer(digitalHumanName: str):
    res_log_dir = './_debug/res/'
    os.makedirs(res_log_dir, mode=0o777, exist_ok=True)
    result_log_path = os.path.join(res_log_dir, 'result.txt')
    # log_status(result_log_path, "running|!")

    infer_log_dir = './_debug/logs/'
    os.makedirs(infer_log_dir, mode=0o777, exist_ok=True)
    infer_file_path = os.path.join(infer_log_dir, 'test.txt')
    logger = setup_logger(digitalHumanName, infer_file_path)

    process = inferring_processes.get(digitalHumanName)
    if process:
        try:
            logger.info(f"Terminating process for {digitalHumanName}")
            process.terminate()
            process.wait(timeout=10)
            del inferring_processes[digitalHumanName]
            log_status(result_log_path, f"fail\n")
            return f"Process for {digitalHumanName} terminated successfully."
        except subprocess.TimeoutExpired:
            logger.error(f"Process for {digitalHumanName} did not terminate in time. Force killing.")
            process.kill()
            del inferring_processes[digitalHumanName]
            log_status(result_log_path, f"fail\n")
            return f"Process for {digitalHumanName} forcefully killed."
        except Exception as e:
            logger.error(f"Error terminating process for {digitalHumanName}: {e}")
            return f"Error terminating process: {e}"
    else:
        return f"No running process found for {digitalHumanName}"

def video_add_audio(video_path: str, audio_path: str, output_dir: str, digitalHumanName, testAudioName, infer_file_path):
    _ext_video = os.path.basename(video_path).strip().split('.')[-1]
    _ext_audio = os.path.basename(audio_path).strip().split('.')[-1]
    if _ext_audio not in ['mp3', 'wav']:
        raise Exception('audio format not support')
    _codec = 'copy'
    if _ext_audio == 'wav':
        _codec = 'aac'
    result = os.path.join(output_dir, '{}.{}'.format(digitalHumanName + '_talk_' + testAudioName, _ext_video))
    ff = FFmpeg(
        inputs={video_path: None, audio_path: None},
        outputs={result: '-y -map 0:v -map 1:a -c:v copy -c:a {} -shortest -threads 4'.format(_codec)})
    # print(ff.cmd)
    with open(infer_file_path, 'a') as log_file:
        ff.run(stdout=log_file, stderr=log_file)
    # ff.run()
    return result

# unknow error stopped
def start_inference(task_id, digitalHumanName, testAudioName, inference_part, publicId, status_queue):
    
    shm = shared_memory.SharedMemory(create=True, size=2 * struct.calcsize('i'))
    shm_name = shm.name
    shm.buf[:4] = struct.pack('i', 100**2) # define total_step
    shm.buf[4:8] = struct.pack('i', 100**2) # define current_step

    res_log_dir = './_debug/res/'
    os.makedirs(res_log_dir, mode=0o777, exist_ok=True)
    result_log_path = os.path.join(res_log_dir, 'result.txt')
    log_status(result_log_path, "running|!")

    infer_log_dir = './_debug/logs/'
    os.makedirs(infer_log_dir, mode=0o777, exist_ok=True)
    infer_file_path = os.path.join(infer_log_dir, 'test.txt')
    logger = setup_logger(digitalHumanName, infer_file_path)
    logger.info(f'[---------------Start Processing with PID-{os.getpid()}---------------]')

    try:
        start_time = datetime.now()
        test_audio = f'./inference/audio_inputs/{testAudioName}.wav'
        logger.info('Start audio processing')
        hubert_processor = HubertProcessor()
        # 模拟unkown error ocurred
        # os._exit(1) 
        hubert_processor.process_audio(test_audio, logger)
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        logger.info(f'Finish Audio Processing at cost {elapsed_time}s')
    except Exception as e:
        log_status(result_log_path, f"fail\n")
        logger.exception(f'Error occurred during audio processing: {e}')
        close_logger(logger)
        status_queue.put((task_id, "failed"))
        return

    PUBLILC_ID_DIC: Dict[str, str] = {
        "250220": "wangwenjie",
        "250221": "boyinnv"
    }
    if publicId == None or publicId == '':
        cmd = f'python ./main.py ./data/{digitalHumanName}/ --workspace ./trial/{digitalHumanName}_{inference_part}/ \
            -O --test --test_train --aud ./inference/audio_inputs/{testAudioName}_hu.npy --shm_name {shm_name} --task_id {task_id}'
        command = [
            f"./data/{digitalHumanName}/",
            "--workspace", f"./trial/{digitalHumanName}_{inference_part}/",
            "-O", 
            "--test", 
            "--test_train",
            "--aud", f"./inference/audio_inputs/{testAudioName}_hu.npy",
            "--shm_name", shm_name,
            "--task_id", str(task_id)
        ]
    elif publicId in PUBLILC_ID_DIC:
        public_value = PUBLILC_ID_DIC[publicId]
        cmd = f'python ./main.py ./_public_data/{public_value}/datasets --workspace ./_public_data/{public_value}/trial/{inference_part}/ \
            -O --test --test_train --aud ./inference/audio_inputs/{testAudioName}_hu.npy --shm_name {shm_name} --task_id {task_id}'
        command = [
            f"./_public_data/{public_value}/datasets",
            "--workspace", f"./_public_data/{public_value}/trial/{inference_part}/",
            "-O", 
            "--test", 
            "--test_train",
            "--aud", f"./inference/audio_inputs/{testAudioName}_hu.npy",
            "--shm_name", shm_name,
            "--task_id", str(task_id)
        ]
    else:
        log_status(result_log_path, f"fail\n")
        logger.error(f'Failed to infer, publicId: {publicId} matched error!')
        close_logger(logger)
        status_queue.put((task_id, "failed"))
        return
    
    try:
        with open(infer_file_path, "a") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            main(command)
        logger.info('[---------------Finished Video Inference---------------]')
    except Exception as e:
        log_status(result_log_path, f"fail\n")
        logger.error(f'Failed to Infer, Exception is : {e}')
        status_queue.put((task_id, "failed"))
        return

    if publicId == None or publicId == '':
        output_dir = f'./trial/{digitalHumanName}_{inference_part}/results/{task_id}'
    else:
        output_dir = f'./_public_data/{public_value}/trial/{inference_part}/results/{task_id}'
    output_video = os.path.join(output_dir, f'temp_all.mp4')
    
    try:
        video_add_audio(output_video, test_audio, './inference/video_outputs', digitalHumanName, testAudioName, infer_file_path)
        log_status(result_log_path, f"success\n")
        logger.info('[---------------Finished Video ADD Audio---------------]')
        status_queue.put((task_id, "completed"))
    except Exception as e:
        log_status(result_log_path, f"fail\n")
        logger.info('[---------------Failed to add audio into video---------------]')
        status_queue.put((task_id, "failed"))
    finally:
        safe_delete(output_dir, logger)
        close_logger(logger)

def safe_delete(dir_path, logger):
    if os.path.exists(dir_path):
        try:
            os.chmod(dir_path, 0o777)
            # os.remove(file_path)
            shutil.rmtree(dir_path)
            logger.info(f"Dir {dir_path} has been safely deleted.\n")
        except PermissionError:
            logger.error(f"Permission error: cannot delete {dir_path}. Check your Dir permissions.\n")
        except Exception as e:
            logger.error(f"An error occurred while deleting the Dir: {e}\n")
    else:
        logger.error(f"The Dir {dir_path} does not exist.\n")

def get_infer_progress(digitalHumanName, testAudioName, inference_part):
    
    if digitalHumanName not in inferring_processes:
        return {"error": f"{digitalHumanName} not found in inferring_processes"}
    shm_name = inferring_processes[digitalHumanName]
    
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
    except FileNotFoundError:
        return {"error": f"Shared memory {shm_name} not found"}
    
    try:
        shared_total = struct.unpack('i', shm.buf[:4])[0]
        shared_step = struct.unpack('i', shm.buf[4:8])[0]
    except struct.error:
        return {"error": "Failed to unpack shared memory"}
    
    return shared_total, shared_step

def init_system():
    nvmlInit()
    print("GPU初始化完毕 Starting the system...")
    queue_thread = threading.Thread(target=process_consumer, daemon=True)
    queue_thread.start()
    print("系统初始化完毕 Starting the system...")

def get_gpu_usage():
    try:
        gpu_handle = nvmlDeviceGetHandleByIndex(0)
        gpu_mem_info = nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_used = gpu_mem_info.used / (1024 ** 2)  # 转换为 MB
        gpu_total = gpu_mem_info.total / (1024 ** 2)  # 转换为 MB
        gpu_free = gpu_mem_info.free / (1024 ** 2)  # 转换为 MB，未使用的显存
        gpu_util = gpu_used * 100 / gpu_total  # 使用率
        return gpu_free
    except Exception as e:
        print(f"获取 GPU 使用情况时出错: {e}")
        return -1

def print_process_tree(pid=None, level=0):
    """ 递归打印进程树 """
    if pid is None:
        pid = os.getpid()  # 获取当前进程 ID
    try:
        process = psutil.Process(pid)
        indent = " " * (level * 4)
        print(f"{indent}└── {process.name()} (PID={process.pid})")
        for child in process.children(recursive=True):  # 递归获取子进程
            print_process_tree(child.pid, level + 1)
    except psutil.NoSuchProcess:
        pass

def get_process_usage(pid=None):
    if pid is None:
        pid = psutil.Process().pid  # 默认从当前进程开始

    total_cpu = 0.0
    total_memory = 0.0

    try:
        process = psutil.Process(pid)
        # 获取当前进程的 CPU 使用率和内存使用情况
        total_cpu += process.cpu_percent(interval=0.5)  # interval=1.0 是为了获取正确的 CPU 使用率
        total_memory += process.memory_info().rss / (1024 ** 2)  # 以 MB 为单位

        # 获取当前进程的子进程
        for child in process.children():
            child_cpu, child_memory = get_process_usage(child.pid)  # 递归获取子进程的 CPU 和内存
            total_cpu += child_cpu
            total_memory += child_memory

    except psutil.NoSuchProcess:
        pass
    except psutil.AccessDenied:
        pass
    cpu_cores_usage = total_cpu / TOTAL_CPU_CORES
    return cpu_cores_usage, total_memory

def process_consumer():
    while True:
        time.sleep(1)
        print("\n📌 当前进程树：")
        print_process_tree()
        
        # 处理任务完成的通知，进程间通信
        while not status_update_queue.empty():
            task_id, status = status_update_queue.get(timeout=0.5)
            with task_lock:
                if task_id in tasks:
                    tasks[task_id]["status"] = status  # 更新任务状态
                    process = tasks[task_id].get("process")
                    if status in ["completed", "failed", "terminated"]:
                        if process:
                            if process.is_alive():
                                print(f"⚠️ 任务 {task_id} 仍在运行，强制终止...")
                                process.terminate()
                                process.join()
                            print(f"✅ 任务 {task_id} 的进程已正常退出 (exitcode={process.exitcode}).")
                        del tasks[task_id]
                        gc.collect()
                        print(f"🗑️ 任务 {task_id} 已从任务列表中删除。")
                        
        # 控制最大并发任务数
        with task_lock:
            print('tasks = ', tasks)
            print('pid = ', psutil.Process().pid)
            gpu_free = get_gpu_usage()
            cpu_usage, total_memory  = get_process_usage()
            system_usage = {
                'cpu_usage': cpu_usage,
                'memory_usage': total_memory,
                'gpu_free': gpu_free,
            }
            
            print(f"CPU核心使用: {system_usage['cpu_usage']:.2f}%")
            print(f"RAM内存占用: {system_usage['memory_usage']:.2f}MB")
            print(f"未用显存: {system_usage['gpu_free']:.2f}MB")
            
            running_tasks = sum(1 for tmp in tasks.values() if tmp["status"] == "running")
            if running_tasks >= MAX_WORKERS:
                continue  # 任务已满，不需要获取系统使用情况

            if system_usage['cpu_usage'] > 70.0 or system_usage['memory_usage'] > 12000.0 or system_usage['gpu_free'] < 2000.0:
                continue  # 系统资源占用过高，不启动新任务

        try:
            task_id, digitalHumanName, testAudioName, inference_part, publicId = task_queue.get(timeout=0.5)
        except queue.Empty:
            continue  # 没有任务时继续等待

        with task_lock:
            tasks[task_id] = {"status": "running", "process": None}
            # 创建进程执行任务，并传入 `status_update_queue`
            p = multiprocessing.Process(target=start_inference, \
                                        args=(task_id, digitalHumanName, testAudioName, inference_part, publicId, status_update_queue))
            tasks[task_id]["process"] = p
            print(f"Processing task {task_id}")
            print('🟢 Updated tasks = ', tasks)
        p.start()

def submit_task(digitalHumanName, testAudioName, inference_part, publicId):
    task_id = str(uuid.uuid4())
    with task_lock:
        tasks[task_id] = {"status": "waiting", 
                          "process": None}  # 任务初始化
        print(f"🟢 Added task {task_id}, current tasks: {tasks}")
        task_queue.put((task_id, digitalHumanName, testAudioName, inference_part, publicId))
    return {"task_id": task_id, "status": "waiting"}

def terminate_task(task_id):
    with task_lock:
        if task_id in tasks and tasks[task_id]["status"] == "running":
            process = tasks[task_id]["process"]
            if process and process.is_alive():
                process.terminate()
                process.join()
                tasks[task_id]["status"] = "terminated"
                tasks[task_id]["process"] = None  # 清理进程
                print(f"❌ Task {task_id} has been terminated.")
                return {"task_id": task_id, "status": "terminated"}
        elif task_id in tasks:
            return {"task_id": task_id, "status": tasks[task_id]["status"]}
        else:
            return {"error": "Task not found"}
