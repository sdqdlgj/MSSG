import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile
import argparse

def run(args):
    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    if 'train' in args.input_file_list:
        dataType = 'train'
    else:
        dataType = 'val'
    with open(args.input_file_list, 'r') as fid:
        for file_path in fid:
            file_path = file_path.strip("\n")
            print(file_path)
            outDir = os.path.join(args.out_path, dataType)
            df = pandas.read_csv(file_path, header=None)
            new_headers = ['video_id',	'frame_timestamp',	'entity_box_x1',	'entity_box_y1',	'entity_box_x2',	'entity_box_y2',	'label',	'entity_id',	'label_id',	'instance_id']
            df.columns = new_headers
            dfNeg = pandas.concat([df[df['label_id'] == 0], df[df['label_id'] == 2]])
            dfPos = df[df['label_id'] == 1]
            insNeg = dfNeg['instance_id'].unique().tolist()
            insPos = dfPos['instance_id'].unique().tolist()
            df = pandas.concat([dfPos, dfNeg]).reset_index(drop=True)
            df = df.sort_values(['entity_id', 'frame_timestamp']).reset_index(drop=True)
            entityList = df['entity_id'].unique().tolist()
            df = df.groupby('entity_id')
            for l in df['video_id'].unique().tolist():
                d = os.path.join(outDir, l[0])
                if not os.path.isdir(d):
                    os.makedirs(d)
            for entity in entityList:
                insData = df.get_group(entity)
                videoKey = insData.iloc[0]['video_id']
                entityID = insData.iloc[0]['entity_id']
                videoDir = os.path.join(args.AVADataPath, 'orig_videos', dic[dataType])
                videoFile = glob.glob(os.path.join(videoDir, '{}.*'.format(videoKey)))[0]
                V = cv2.VideoCapture(videoFile)
                insDir = os.path.join(os.path.join(outDir, videoKey, entityID))
                if not os.path.isdir(insDir):
                    os.makedirs(insDir)
                j = 0
                for _, row in insData.iterrows():
                    imageFilename = os.path.join(insDir, str("%.2f"%row['frame_timestamp'])+'.jpg')
                    V.set(cv2.CAP_PROP_POS_MSEC, row['frame_timestamp'] * 1e3)
                    _, frame = V.read()
                    h = numpy.size(frame, 0)
                    w = numpy.size(frame, 1)
                    x1 = row['entity_box_x1']
                    y1 = row['entity_box_y1']
                    x2 = row['entity_box_x2']
                    y2 = row['entity_box_y2']
                    new_x1 = x1 - 0.75*(x2-x1)
                    new_x2 = x2 + 0.75*(x2-x1)
                    if new_x1 < 0:
                        new_x1 = 0
                    if new_x2 > 1:
                        new_x2 = 1
                    new_y1 = y1
                    new_y2 = y2 + 3*(y2-y1)
                    if new_y2 > 1:
                        new_y2 = 1
                    facebody = frame[int(new_y1*h):int(new_y2*h), int(new_x1*w):int(new_x2*w), :]
                    j = j+1
                    cv2.imwrite(imageFilename, facebody)
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--AVADataPath', type=str, dest='AVADataPath', default='/share/home/liguanjun/data/AVA', help="AVADataPath")
    parser.add_argument('--out_path', type=str, dest='out_path', default='/share/home/liguanjun/data/AVA/clips_videos_test', help="out_path")
    parser.add_argument('--input_file_list', type=str, dest='input_file_list', default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/facebody_process/split/train_file_list.26', help="input_file_list")
    args = parser.parse_args()
    run(args)

