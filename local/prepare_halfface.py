import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile
import argparse
import tqdm
import csv


def get_all_files(directory):
    all_files = []  
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(file)
    return all_files

def run(args):
    pre_clip_name = 'xxx'
    with open(args.input_file_list, 'r') as fid:
        for file_path in fid:
            print(file_path)
            file_path = file_path.strip("\n")
            # list all dirs:
            files = get_all_files(file_path)
            for file in files:
                down_file_path = os.path.join(args.down_out_path,  "/".join(file_path.split('/')[-3:]))
                file_name = file
                if not os.path.exists(down_file_path):
                    os.makedirs(down_file_path)
                face = cv2.imread(os.path.join(file_path, file))
                h, w, _ = face.shape
                down_face = face[int(h/2):,:]
                cv2.imwrite(os.path.join(down_file_path, file_name),down_face)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--AVADataPath', type=str, dest='AVADataPath', default='/share/home/liguanjun/data/AVA', help="AVADataPath")
    parser.add_argument('--down_out_path', type=str, dest='down_out_path', default='/share/home/liguanjun/data/AVA/clips_videos_background_resize_test/down', help="out_path")
    parser.add_argument('--input_file_list', type=str, dest='input_file_list', default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/face_process/split/val_file_list.4', help="AVADataPath")
    parser.add_argument('--size', type=int, dest='size', default=112, help="size")
    parser.add_argument('--csv_split_root', type=str, dest='csv_split_root', default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/csv_split/val', help="AVADataPath")
    args = parser.parse_args()
    run(args)