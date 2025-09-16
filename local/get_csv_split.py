import time, os, sys, torch, argparse, warnings, glob
import tqdm
import random
import numpy as np
import csv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Model Training")
    parser.add_argument('--dataPathAVA',  type=str, default="/share/home/liguanjun/data/AVA", help='Save path of AVA dataset')
    parser.add_argument('--out_path',     type=str, default="/share/home/liguanjun/src/LGJ_PretrainASD/predata/csv_split")
    args = parser.parse_args()

    for name in ['train', 'val']:
        os.makedirs(os.path.join(args.out_path, name), exist_ok=True)
        cur_csvfile = os.path.join(args.dataPathAVA, 'csv', name+'_orig.csv')
        with open(cur_csvfile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            i = -1
            clip_info = {}
            pre_video_name = 'uuu'
            for row in reader:
                i += 1
                if i == 0:
                    continue
                clip_name = row[7]
                video_name = clip_name[:11]
                if video_name != pre_video_name:
                    if pre_video_name != 'uuu':
                        csv_fid.close()
                    print(video_name)
                    csv_fid = open(os.path.join(args.out_path, name, video_name+'.csv'), mode='w', newline='')
                    writer = csv.writer(csv_fid, delimiter=',')
                writer.writerow(row)
                pre_video_name = video_name
        # 		if clip_name != pre_clip_name:
        # 			clip_info[clip_name] = [row[1]]
        # 		else:
        # 			clip_info[clip_name].append(row[1])
        # 		pre_clip_name = clip_name
        # 		pre_video_name = video_name
        # torch.save(clip_info, os.path.join(args.out_path, name+'_clip_info.pt'))
