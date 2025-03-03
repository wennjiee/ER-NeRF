import os
import uuid
import glob
from ffmpy import FFmpeg
import sys
import logging
import subprocess
from datetime import datetime
from multiprocessing import shared_memory
import struct
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# print(os.getcwd())
from data_utils.hubert_processor import HubertProcessor
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=2)
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

def run_subprocess(cmd, log_file_path, result_log_path, logger, digitalHumanName):
    try:
        logger.info(f"Start inference for \ncommand = {cmd}")
        file_path = os.path.abspath(log_file_path)
        with open(file_path, "a") as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True, bufsize=1)
            inferring_processes[digitalHumanName] = process
            process.wait()
        if process.returncode == 0:
            logger.info(f' ===== Finish Inference Successfully =====')
            return 0
        else:
            log_status(result_log_path, f"fail\n")
            logger.error(f'Failed to Infer with return code {process.returncode}')
            return -1
    except Exception as e:
        log_status(result_log_path, f"fail\n")
        logger.error(f'Failed to Infer, Exception is : {e}')
        return -2

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

def start(taskId, digitalHumanName, testAudioName, inference_part, publicId=None):
    
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
    logger.info('[---------------Start Inferring---------------]')

    try:
        hubert_processor = HubertProcessor()
        start_time = datetime.now()
        test_audio = f'./inference/audio_inputs/{testAudioName}.wav'
        logger.info('Start process audio')
        hubert_processor.process_audio(test_audio, logger)
        end_time = datetime.now()
        elapsed_time = (end_time - start_time).total_seconds()
        logger.info(f'Finish Audio Processing at cost {elapsed_time}s')
    except Exception as e:
        log_status(result_log_path, f"fail\n")
        logger.exception(f'Error during audio processing: {e}')
        close_logger(logger)
        return
    
    public_id_dic: Dict[str, str] = {
        "250220": "wangwenjie",
        "250221": "liuchang",
        "250222": "boyinnv"
    }
    if publicId == None or publicId == '':
        cmd = f'python ./main.py ./data/{digitalHumanName}/ --workspace ./trial/{digitalHumanName}_{inference_part}/ \
            -O --test --test_train --aud ./inference/audio_inputs/{testAudioName}_hu.npy --shm_name {shm_name} --task_id {taskId}'
    elif publicId in public_id_dic:
        public_value = public_id_dic[publicId]
        cmd = f'python ./main.py ./_public_data/{public_value}/datasets --workspace ./_public_data/{public_value}/trial/{inference_part}/ \
    -O --test --test_train --aud ./inference/audio_inputs/{testAudioName}_hu.npy --shm_name {shm_name} --task_id {taskId}'
    else:
        log_status(result_log_path, f"fail\n")
        logger.error(f'Failed to Infer, publicId: {publicId} matched error!')
        close_logger(logger)
        return
    
    status = run_subprocess(cmd, infer_file_path, result_log_path, logger, digitalHumanName)
    
    if status == 0:
        if publicId == None:
            output_video = os.path.join(f'./trial/{digitalHumanName}_{inference_part}/results/{taskId}.mp4')
            # result_paths = sorted(glob.glob(os.path.join(f'./trial/{digitalHumanName}_{inference_part}/results/', '*.mp4')))
        else:
            output_video = os.path.join(f'./_public_data/{public_value}/trial/{inference_part}/results/{taskId}.mp4')
            # result_paths = sorted(glob.glob(os.path.join(f'./_public_data/{public_value}/trial/{inference_part}/results/', '*.mp4')))
        video_add_audio(output_video, test_audio, './inference/video_outputs', digitalHumanName, testAudioName, infer_file_path)
        log_status(result_log_path, f"success\n")
        logger.info('[---------------Finished Video ADD Audio---------------]\n')
        safe_delete(output_video, logger)
    close_logger(logger)

def run_infer(digitalHumanName, testAudioName, inference_part, publicId=None):
    taskId = str(uuid.uuid4())
    executor.submit(start, taskId, digitalHumanName, testAudioName, inference_part, publicId)
    return taskId

def safe_delete(file_path, logger):
    if os.path.exists(file_path):
        try:
            os.chmod(file_path, 0o777)
            os.remove(file_path)
            logger.info(f"File {file_path} has been safely deleted.")
        except PermissionError:
            logger.error(f"Permission error: cannot delete {file_path}. Check your file permissions.")
        except Exception as e:
            logger.error(f"An error occurred while deleting the file: {e}")
    else:
        logger.error(f"The file {file_path} does not exist.")

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