import os
import glob
import sys
import logging
import subprocess

log_file_path = './logs/train_defualt.log'
def setup_logging():
    global log_file_path
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler()
        ]
    )

def run_command(cmd):
    logging.info(f"Running command: {cmd}")
    try:
        with open(log_file_path, "a") as log_file:
            process = subprocess.Popen(cmd, shell=True, stdout=log_file, stderr=log_file, text=True, bufsize=1)
            process.wait()
        if process.returncode == 0:
            logging.info(f' ===== extracted audio labels =====')
        else:
            logging.error(f'Failed to extract audio features with return code {process.returncode}')
    except Exception as e:
        logging.error(f'Failed to extract audio features: {e}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("[ERROR] Missing arguments: train_name and log_file_path")
        sys.exit(1)

    train_name = sys.argv[1]
    log_file_path = sys.argv[2]
    setup_logging()
    logging.info('Starting in Train.py')

    cmd0 = f'python data_utils/process.py data/{train_name}/{train_name}.mp4 --log_file {log_file_path}'
    cmd1 = f'python main.py data/{train_name}/ --workspace trial/{train_name}_head/ -O --iters 100000'
    cmd2 = f'python main.py data/{train_name}/ --workspace trial/{train_name}_head/ -O --iters 125000 --finetune_lips --patch_size 32'

    preprocessed_data = f'data/{train_name}/transforms_train.json'
    if not os.path.exists(preprocessed_data):
        logging.info(f"Preprocessed data not found, running preprocessing: {cmd0}")
        run_command(cmd0)

    logging.info("******************start training head******************")
    run_command(cmd1)
    logging.info("******************finished training head******************")

    logging.info("******************start finetune lips******************")
    run_command(cmd2)
    logging.info("******************finished finetune******************")

    checkpoints_paths = sorted(glob.glob(os.path.join(f'trial/{train_name}_head/checkpoints/', '*.pth')), reverse=True)
    ck_path = checkpoints_paths[0] if checkpoints_paths else None
    if not ck_path:
        logging.error("No checkpoint found. Exiting.")
        sys.exit(1)

    cmd3 = f'python main.py data/{train_name}/ --workspace trial/{train_name}_torso/ -O --torso --head_ckpt {ck_path} --iters 200000'
    logging.info("******************start training torso******************")
    run_command(cmd3)
    logging.info("******************finished training torso******************")
