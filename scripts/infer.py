import os
import uuid
import glob
from ffmpy import FFmpeg
import sys
sys.path.append(os.getcwd()) # most important !!!
from data_utils import process

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
        outputs={result: '-map 0:v -map 1:a -c:v copy -c:a {} -shortest'.format(_codec)})
    print(ff.cmd)
    ff.run()
    return result

if __name__ == '__main__':
    digitalHumanName = 'lc_128'
    testAudioName = 'LC_Vocals_16'
    inference_part = 'head'
    checkpoints_paths = sorted(glob.glob(os.path.join(f'trial/{digitalHumanName}_{inference_part}/checkpoints/', '*.pth')), reverse=True)
    ck_path = checkpoints_paths[0].replace('\\', '/')
    test_audio = f'inference/audio_inputs/{testAudioName}.wav'
    if not os.path.exists(test_audio.replace('.wav', '_hu.npy')):
        process.extract_audio_features(test_audio, mode='hubert')
    cmd = f'python main.py data/{digitalHumanName}/ --workspace trial/{digitalHumanName}_{inference_part}/ -O --test --test_train --aud inference/audio_inputs/{testAudioName}_hu.npy'
    os.system(cmd)
    print('cmd processed')
    result_paths = sorted(glob.glob(os.path.join(f'trial/{digitalHumanName}_{inference_part}/results/', '*.mp4')))
    output_video = result_paths[0].replace('\\', '/')
    video_add_audio(output_video, test_audio, './inference/video_outputs', digitalHumanName, testAudioName)