import time, os, torch, argparse, warnings, glob

from dataLoader import train_loader, val_loader
from utils.tools import *
from ASD import ASD
import random
import numpy as np
from torch.utils.data import Subset

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True  # 确保每次卷积运算的确定性
    torch.backends.cudnn.benchmark = False     # 禁用cuDNN的自动优化
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)

def main():
    set_seed(111)
    # This code is modified based on this [repository](https://github.com/TaoRuijie/TalkNet-ASD).
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description = "Model Training")
    # Training details
    parser.add_argument('--lr',           type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lrDecay',      type=float, default=0.95,  help='Learning rate decay rate')
    parser.add_argument('--maxEpoch',     type=int,   default=30,    help='Maximum number of epochs')
    parser.add_argument('--testInterval', type=int,   default=1,     help='Test and save every [testInterval] epochs')
    parser.add_argument('--batchSize',    type=int,   default=1000,  help='Dynamic batch size, default is 2000 frames')
    parser.add_argument('--nDataLoaderThread', type=int, default=64,  help='Number of loader threads')
    parser.add_argument('--small_training',    type=int,   default=1,  help='small training')
    parser.add_argument('--small_trainset',    type=int,   default=1,  help='small training trainset')
    parser.add_argument('--small_valset',    type=int,   default=10,  help='small training valset')
    parser.add_argument('--train_type',    type=str,   default='face',  help='train_type')
    parser.add_argument('--background_H',    type=int,   default='224',  help='background_H')
    parser.add_argument('--encoder_struct',    type=str,   default='32,64,128',  help='encoder_struct')
    # Data path
    parser.add_argument('--dataPathAVA',  type=str, default="/data/data/user/lgj/data/AVA", help='Save path of AVA dataset')
    parser.add_argument('--savePath',     type=str, default="exps/exp1")
    # Data selection
    parser.add_argument('--evalDataType', type=str, default="val", help='Only for AVA, to choose the dataset for evaluation, val or test')
    # For download dataset only, for evaluation only
    parser.add_argument('--downloadAVA',     dest='downloadAVA', action='store_true', help='Only download AVA dataset and do related preprocess')
    parser.add_argument('--evaluation',      dest='evaluation', action='store_true', help='Only do evaluation by using pretrained model [pretrain_AVA_CVPR.model]')
    args = parser.parse_args()
    # Data loader
    args = init_args(args)

    if args.downloadAVA == True:
        preprocess_AVA(args)
        quit()

    print(f'Using {torch.cuda.device_count()} GPUs')

    if args.train_type == 'background':
        args.batchSize = 800
        H = args.background_H
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_background_resize_224')
    elif args.train_type == 'face_large':
        H = 112
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_face_large_region')
    elif args.train_type == 'face_small':
        H = 112
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_face_small_region')
    elif args.train_type == 'face_left':
        H = 112
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_left_face')
    elif args.train_type == 'face_right':
        H = 112
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_right_face')
    elif args.train_type == 'face_up':
        H = 112
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_up_face')
    elif args.train_type == 'face_down':
        H = 112
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_down_face')
    elif args.train_type == 'face_body':
        args.batchSize = 800
        H = 224
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_face_body')  
    elif args.train_type == 'face_body_large':
        args.batchSize = 800
        H = 224
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_face_body_large')  
    elif args.train_type == 'full':
        args.batchSize = 800
        H = 224
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_full_frame')
    elif args.train_type == 'full_main':
        args.batchSize = 800
        H = 224
        args.visualPathAVA  = os.path.join(args.dataPathAVA, 'clips_videos_full_frame_mainspeaker_224')
    else:
        if args.visualPathAVA != 'face':
            print("Wrong train_type!")
            return 
        H = 112
    encoder_struct = args.encoder_struct.split(",")
    args.encoder_struct = [int(item) for item in encoder_struct]
    print(args)
    if args.small_training == 0: 
        loader = train_loader(trialFileName = args.trainTrialAVA, \
                            audioPath      = os.path.join(args.audioPathAVA , 'train'), \
                            visualPath     = os.path.join(args.visualPathAVA, 'train'), \
                            H = H,
                            **vars(args))
        trainLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread, pin_memory = True)

        loader = val_loader(trialFileName = args.evalTrialAVA, \
                            audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
                            visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
                            H = H,
                            **vars(args))
        valLoader = torch.utils.data.DataLoader(loader, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread, pin_memory = False)
    else:
        train_dataset = train_loader(trialFileName = args.trainTrialAVA, \
                            audioPath      = os.path.join(args.audioPathAVA , 'train'), \
                            visualPath     = os.path.join(args.visualPathAVA, 'train'), \
                            H = H,
                            **vars(args))
        val_dataset = val_loader(trialFileName = args.evalTrialAVA, \
                            audioPath     = os.path.join(args.audioPathAVA , args.evalDataType), \
                            visualPath    = os.path.join(args.visualPathAVA, args.evalDataType), \
                            H = H,
                            **vars(args))
        train_indices = range(0, args.small_trainset)  # 加载前200个样本
        val_indices = range(0, args.small_valset)  # 加载前1000个样本
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        trainLoader = torch.utils.data.DataLoader(train_subset, batch_size = 1, shuffle = True, num_workers = args.nDataLoaderThread, pin_memory = True)
        valLoader = torch.utils.data.DataLoader(val_subset, batch_size = 1, shuffle = False, num_workers = args.nDataLoaderThread, pin_memory = False)

    # if args.evaluation == True:
    #     s = ASD(**vars(args))
    #     s.loadParameters('weight/pretrain_AVA_CVPR.model')
    #     print("Model %s loaded from previous state!"%('pretrain_AVA_CVPR.model'))
    #     mAP = s.evaluate_network(loader = valLoader, **vars(args))
    #     print("mAP %2.2f%%"%(mAP))
    #     quit()

    modelfiles = glob.glob('%s/model_0*.model'%args.modelSavePath)
    modelfiles.sort()  
    if len(modelfiles) >= 1:
        print("Model %s loaded from previous state!"%modelfiles[-1])
        epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
        s = ASD(epoch = epoch, **vars(args))
        s.loadParameters(modelfiles[-1])
    else:
        epoch = 1
        s = ASD(epoch = epoch, **vars(args))

    mAPs = []
    scoreFile = open(args.scoreSavePath, "a+")

    while(1):    
        # loss, lr = 0,0    
        loss, lr = s.train_network(epoch = epoch, loader = trainLoader, type=args.train_type, **vars(args))
        
        if epoch % args.testInterval == 0:        
            s.saveParameters(args.modelSavePath + "/model_%04d.model"%epoch)
            mAPs.append(s.evaluate_network(epoch = epoch, loader = valLoader, type=args.train_type, **vars(args)))
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, mAP %2.2f%%, bestmAP %2.2f%%"%(epoch, mAPs[-1], max(mAPs)))
            scoreFile.write("%d epoch, LR %f, LOSS %f, mAP %2.2f%%, bestmAP %2.2f%%\n"%(epoch, lr, loss, mAPs[-1], max(mAPs)))
            scoreFile.flush()

        if epoch >= args.maxEpoch:
            quit()

        epoch += 1

if __name__ == '__main__':
    main()
