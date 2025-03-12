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


shm = shared_memory.SharedMemory(create=True, size=2 * struct.calcsize('i'))
shm_name = shm.name
shm.buf[:4] = struct.pack('i', 100**2) # define total_step
shm.buf[4:8] = struct.pack('i', 100**2) # define current_step
digitalHumanName = 'lc10s'
inference_part = 'torso'
testAudioName = 'yilian'
task_id = '250312'

command = [
    f"./data/{digitalHumanName}/",
    "--workspace", f"./trial/lc_128_torso/",
    "-O", "--test", "--test_train",
    "--aud", f"./inference/audio_inputs/{testAudioName}_hu.npy",
    "--shm_name", shm_name,
    "--task_id", str(task_id)
]
main(command)