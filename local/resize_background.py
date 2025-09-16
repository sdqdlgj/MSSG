import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile
import argparse
import tqdm

def get_all_files(directory):
    all_files = []  
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(file)
    return all_files

def run(args):
    with open(args.input_file_list, 'r') as fid:
        for file_path in fid:
            print(file_path)
            file_path = file_path.strip("\n")
            # list all dirs:
            files = get_all_files(file_path)
            for file in files:
                new_file_path = os.path.join(args.out_path,  "/".join(file_path.split('/')[-3:]))
                file_name = file
                if not os.path.exists(new_file_path):
                    os.makedirs(new_file_path)
                back = cv2.imread(os.path.join(file_path, file))
                back = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY)
                back = cv2.resize(back, (args.size,args.size))
                cv2.imwrite(os.path.join(new_file_path, file_name),back)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--AVADataPath', type=str, dest='AVADataPath', default='/share/home/liguanjun/data/AVA', help="AVADataPath")
    parser.add_argument('--out_path', type=str, dest='out_path', default='/share/home/liguanjun/data/AVA/clips_videos_background_resize_test', help="out_path")
    parser.add_argument('--input_file_list', type=str, dest='input_file_list', default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/background_process/split/val_file_list.54', help="AVADataPath")
    parser.add_argument('--size', type=int, dest='size', default=224, help="size")
    args = parser.parse_args()
    run(args)