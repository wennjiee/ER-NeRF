import os
if __name__ == '__main__':
    # test on the test split
    test_name = 'zhouyuzhu'
    print("******************start testing******************")
    cmd1 = f'python main.py data/{test_name}/ --workspace trial/{test_name}_head/ -O --test' # only render the head and use GT image for torso
    cmd2 = f'python main.py data/{test_name}/ --workspace trial/{test_name}_torso/ -O --torso --test' # render both head and torso
    os.system(cmd1)
    os.system(cmd2)
    print("******************finished testing******************")