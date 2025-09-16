import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile
import argparse
import tqdm

def run(args):
    dir_list = []
    pre_clip_name = 'xxx'
    i = 0
    with open(args.out_file, 'w') as fid:
	    for root, dirs, files in os.walk(args.AVADataPath):
	        if not dirs:
		        clip_name = root.split("/")[-2]
		        if clip_name != pre_clip_name:
		        	i += 1
		        	print(f"{i}:{clip_name}")
		        	pre_clip_name = clip_name
        		fid.write(root+"\n")
        # a = 1





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--AVADataPath', type=str, dest='AVADataPath', default='/data/data/user/lgj/data/AVA/train', help="AVADataPath")
    parser.add_argument('--out_file', type=str, dest='out_file', default='./predata/train_list', help="out_path")
    args = parser.parse_args()
    run(args)
