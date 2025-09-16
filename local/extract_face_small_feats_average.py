import time, os, sys, torch, argparse, warnings, glob
custom_lib_path = os.path.abspath('./')
sys.path.append(custom_lib_path)
custom_lib_path = os.path.abspath('../')
sys.path.append(custom_lib_path)
from dataLoader import feat_extract_loader
import tqdm
# from utils.tools import *
from ASD import ASD
import random
import numpy as np
import csv
import torch 
from sklearn.metrics import average_precision_score

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保每次卷积运算的确定性
    torch.backends.cudnn.benchmark = False     # 禁用cuDNN的自动优化
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Model Training")
    parser.add_argument('--background_H',    type=int,   default='112',  help='background_H')
    parser.add_argument('--dataPathAVA',  type=str, default="/share/home/liguanjun/data/AVA", help='Save path of AVA dataset')
    parser.add_argument('--out_path',     type=str, default="/share/home/liguanjun/src/LGJ_PretrainASD/predata/feats_info")
    parser.add_argument('--face_model_path',     type=str, default="/share/home/liguanjun/src/LGJ_PretrainASD/predata/pretrain_model/face_test")
    parser.add_argument('--clip_info_path',     type=str, default="/share/home/liguanjun/src/LGJ_PretrainASD/predata/clip_info")
    args = parser.parse_args()
    
    # set_seed(111)
    print(f'Using {torch.cuda.device_count()} GPUs')

    train_loader = feat_extract_loader(trialFileName = os.path.join(args.dataPathAVA, 'csv', 'train_loader.csv'), \
			                  audioPath     = os.path.join(args.dataPathAVA, 'clips_audios_aug', 'train'), \
			                  visualPath    = os.path.join(args.dataPathAVA, 'clips_videos_face_small_region', 'train'), \
			                  H = 112,
                              type='train',
			                  )
    val_loader = feat_extract_loader(trialFileName = os.path.join(args.dataPathAVA, 'csv', 'val_loader.csv'), \
		                  	audioPath     = os.path.join(args.dataPathAVA, 'clips_audios', 'val'), \
			                visualPath    = os.path.join(args.dataPathAVA, 'clips_videos_face_small_region', 'val'), \
		                    H = 112,
                            type='val',
		                    )
    trainLoader = torch.utils.data.DataLoader(train_loader, batch_size = 1, shuffle = False, num_workers = 40, pin_memory = False)
    valLoader = torch.utils.data.DataLoader(val_loader, batch_size = 1, shuffle = False, num_workers = 40, pin_memory = False)
    
    s = ASD(encoder_struct=[32,64,128])
    s.loadParameters_multi(args.face_model_path)
    s.eval()
    
    for name in ['val', 'train']:
    # for name in ['val']:
    # for name in ['train']:
        if name == 'train':
            dataloader = trainLoader
        else:
            dataloader = valLoader
        clip_info = torch.load(os.path.join(args.clip_info_path, name+'_clip_info.pt'))
        feat_save_path = os.path.join(args.out_path, name)
        os.makedirs(feat_save_path, exist_ok=True)
        pre_video_name = 'uuu'
        predScores = []
        target_total = []
        soft_total = []
        for item in tqdm.tqdm(dataloader):
            clip_name = item[0][0]
            video_name = clip_name[:11]
            clip_timestamps = clip_info[clip_name]
            audio_feats = item[1][0]
            visual_feats = item[2][0]
            labels = item[3][0]
            with torch.no_grad(): 
                outsAV, outsV, outsA = s.model(audio_feats.cuda(), visual_feats.cuda())
                labels = labels.reshape((-1)).cuda() 
                _, predScore, _, _ = s.lossAV(outsAV, labels)  
            scores = predScore[:, 1].tolist()
            soft_total.extend(scores)
            target_total.extend(labels[:].tolist())
            
            if video_name != pre_video_name:
                if pre_video_name != 'uuu':
                    csv_fid.close()
                print(video_name)
                csv_fid = open(os.path.join(args.out_path, name, video_name+'.csv'), mode='w', newline='')
                writer = csv.writer(csv_fid, delimiter=',')
            pre_video_name = video_name
            ii = -1
            feat_av = outsAV.cpu().numpy()
            feat_v = outsV.cpu().numpy()
            feat_a = outsA.cpu().numpy()
            
            assert feat_av.shape[0] == len(clip_timestamps)
            for timestamp in clip_timestamps:
                ii += 1
                title = clip_name + ":" + timestamp
                cur_feat_av = list(feat_av[ii,:])
                cur_feat_v = list(feat_v[ii,:])
                cur_feat_a = list(feat_a[ii,:])
                writer.writerow([title,cur_feat_av,cur_feat_v,cur_feat_a])
            
        csv_fid.close()    
        mAP = average_precision_score(target_total, soft_total) * 100
        print(mAP)