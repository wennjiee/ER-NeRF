import os
import uuid
import glob
from ffmpy import FFmpeg
import sys
import argparse
from data_utils import process
import logging
import subprocess

def run_command(cmd, args):
    try:
        file_path = os.path.abspath(args.log_file)
        logging.info(f"Running command: {cmd}")
        with open(file_path, "a") as log_file:
            process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, text=True, bufsize=1)
            process.wait()
        if process.returncode == 0:
            logging.info(f' ===== Infering Successfully =====')
        else:
            logging.error(f'Failed to Infer with return code {process.returncode}') # Todo
    except Exception as e:
        logging.error(f'Failed to Infer: {e}') # Todo

def video_add_audio(video_path: str, audio_path: str, output_dir: str, digitalHumanName, testAudioName):
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
        outputs={result: '-y -map 0:v -map 1:a -c:v copy -c:a {} -shortest'.format(_codec)})
    print(ff.cmd)
    ff.run()
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference process for digital human.")
    parser.add_argument('--digitalHumanName', default='lc_128', help="Name of the digital human model.")
    parser.add_argument('--testAudioName', default='LC_Vocals_16', help="Name of the test audio file.")
    parser.add_argument('--inference_part', default='head', help="Part for inference (e.g., 'head').")
    parser.add_argument('--log_file', default=f'', help="Part for inference (e.g., 'head').")
    args = parser.parse_args()
    
    log_file_path = args.log_file
    digitalHumanName = args.digitalHumanName
    testAudioName = args.testAudioName
    inference_part = args.inference_part
    checkpoints_paths = sorted(glob.glob(os.path.join(f'./trial/{digitalHumanName}_{inference_part}/checkpoints/', '*.pth')), reverse=True)
    ck_path = checkpoints_paths[0].replace('\\', '/')
    test_audio = f'./inference/audio_inputs/{testAudioName}.wav'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path, mode='a'),
            ]
    )
    if not os.path.exists(test_audio.replace('.wav', '_hu.npy')):
        process.extract_audio_features(test_audio, mode='hubert', log_file_path=log_file_path)
    cmd = f'python ./main.py ./data/{digitalHumanName}/ --workspace ./trial/{digitalHumanName}_{inference_part}/ \
        -O --test --test_train --aud ./inference/audio_inputs/{testAudioName}_hu.npy'
    run_command(cmd, args)
    logging.info('====== cmd processed ======')
    result_paths = sorted(glob.glob(os.path.join(f'./trial/{digitalHumanName}_{inference_part}/results/', '*.mp4')))
    output_video = result_paths[0].replace('\\', '/')
    video_add_audio(output_video, test_audio, './inference/video_outputs', digitalHumanName, testAudioName)