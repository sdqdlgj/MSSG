import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile
import argparse

def run(args):
    dic = {'train':'trainval', 'val':'trainval', 'test':'test'}
    for dataType in ['train', 'val']:
        print(f'Deal with {dataType}...')
        outDir = os.path.join(args.out_path, dataType)
        df = pandas.read_csv(os.path.join(args.AVADataPath, 'csv', '%s_orig.csv'%(dataType)))
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
        for entity in tqdm.tqdm(entityList, total = len(entityList)):
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
                x1 = int(row['entity_box_x1'] * w)
                y1 = int(row['entity_box_y1'] * h)
                x2 = int(row['entity_box_x2'] * w)
                y2 = int(row['entity_box_y2'] * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=cv2.FILLED)
                j = j+1
                cv2.imwrite(imageFilename, frame)
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--AVADataPath', type=str, dest='AVADataPath', default='/data/data/user/lgj/data/AVA', help="AVADataPath")
    parser.add_argument('--out_path', type=str, dest='out_path', default='/data/data/user/lgj/data/AVA/clips_videos_background', help="out_path")
    args = parser.parse_args()
    run(args)

