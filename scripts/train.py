import os
import glob
if __name__ == '__main__':
    train_name = 'zxy'
    cmd1 = f'python main.py data/{train_name}/ --workspace trial/{train_name}_head/ -O --iters 100000'
    cmd2 = f'python main.py data/{train_name}/ --workspace trial/{train_name}_head/ -O --iters 125000 --finetune_lips --patch_size 32'
    print("******************start training head******************")
    os.system(cmd1)
    print("******************finished training head******************")
    print("******************start finetune lips******************")
    os.system(cmd2)
    print("******************finished finetune******************")
    checkpoints_paths = sorted(glob.glob(os.path.join(f'trial/{train_name}_head/checkpoints/', '*.pth')), reverse=True)
    ck_path = checkpoints_paths[0]
    cmd3 = f'python main.py data/{train_name}/ --workspace trial/{train_name}_torso/ -O --torso --head_ckpt {ck_path} --iters 200000'
    print("******************start training torso******************")
    os.system(cmd3)
    print("******************finished training torso******************")