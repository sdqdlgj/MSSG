import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile
import argparse
import pickle
def run(args):
    videoDir = os.path.join(args.AVADataPath, 'orig_videos', 'trainval')
    videoFile = glob.glob(os.path.join(videoDir,'*.*'))
    size_dic = {}
    for cur_video in videoFile:
        video_name = cur_video.split('/')[-1].split(".")[0]
        print(video_name)
        V = cv2.VideoCapture(cur_video)
        V.set(cv2.CAP_PROP_POS_MSEC, -0 * 1e3)
        _, frame = V.read()
        h = numpy.size(frame, 0)
        w = numpy.size(frame, 1)
        size_dic[video_name] = [w,h]
    with open(os.path.join(args.out_path, 'size_dic.pkl'), 'wb') as file:
        pickle.dump(size_dic, file)
        
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--AVADataPath', type=str, dest='AVADataPath', default='/share/home/liguanjun/data/AVA', help="AVADataPath")
    parser.add_argument('--out_path', type=str, dest='out_path', default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/video_size_info', help="out_path")
    args = parser.parse_args()
    run(args)
