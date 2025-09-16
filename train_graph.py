import os
import torch
import numpy as np
import random
import argparse
from model.models_gnn import *
from dataLoader import GraphDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from torch.utils.data import Subset
import warnings
from torch.optim.swa_utils import AveragedModel
from collections import deque
import copy 

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='SPELL')
parser.add_argument('--gpu_id', type=int, default=0, help='which gpu to run the train_val')
parser.add_argument('--numv', type=int, default=2000, help='number of nodes')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--time_edge', type=int, default=0, help='time_edge')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducibility')
parser.add_argument('--num_epoch', type=int, default=60, help='total number of epochs')
parser.add_argument('--graph_path', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/graphs', help='graph_path')


def main():
    print("begin!")
    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    print(args)
    
    result_path = os.path.join('exps', 'graph_training', 'graph_'+str(args.numv))
    os.makedirs(result_path, exist_ok=True)

    Fdataset_train = GraphDataset(args, mode='train')
    Fdataset_val = GraphDataset(args, mode='val')
    train_loader = DataLoader(Fdataset_train, batch_size=args.batch_size, shuffle=True, num_workers=32, pin_memory = True)
    val_loader = DataLoader(Fdataset_val, batch_size=1, shuffle=False, num_workers=32, pin_memory = True)

    device = ('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')    
    model = MSSG()
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2, mode="max", min_lr=1e-6)

    flog = open(os.path.join(result_path, 'log.txt'), mode = 'w')
    max_mAP = -1
    epoch_max = 0
    scheduler.best = 0
    for epoch in range(1, args.num_epoch+1):
        r = 1.3 - 0.02 * (epoch - 1)
        # r = 1
        str_print = '[{:3d}|{:3d}]:'.format(epoch, args.num_epoch)
        loss = train(model, train_loader, device, optimizer, criterion, scheduler, r)
        res = evaluation(model, val_loader, device)
        scheduler.step(res)
        mAP = res
        if mAP > max_mAP:
            max_mAP = mAP
            epoch_max = epoch
            torch.save(model.state_dict(), os.path.join(result_path, 'chckpoint_best.pt'))
        str_print += ' lr: {:.6f}, mAP: {:.4f},\t (max_mAP: {:.4f} at epoch: {})'.format(scheduler._last_lr[0], mAP, max_mAP, epoch_max)
        print (str_print)
        flog.write(str_print+'\n')
        flog.flush()

    flog.close()

def check_grad(params, clip_th, ignore_th):
    befgad = torch.nn.utils.clip_grad_norm_(params, clip_th).cpu()
    return (not np.isfinite(befgad) or (befgad > ignore_th))

def train(model, train_loader, device, optimizer, criterion, scheduler, r):
    model.train()
    loss_ce_sum = 0.
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        output = output / r
        output = F.softmax(output, dim = -1)[:,1]
    
        loss_ce = criterion(output, data.y.reshape((-1)))

        loss = loss_ce

        loss.backward()
        loss_ce_sum += loss_ce.item() if type(loss_ce) != int else 0
        
        if check_grad(model.parameters(), 1, 100000):
            optimizer.zero_grad()
        optimizer.step()

    # scheduler.step()

    return loss_ce_sum/len(train_loader)   


def evaluation(model, val_loader, device):
    model.eval()
    target_total = []
    soft_total = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            data = data.to(device)
            x = data.x
            y = data.y
            output = model(data)
            scores = F.softmax(output, dim = -1)[:,1].tolist()
            
            soft_total.extend(scores)
            target_total.extend(y[:, 0].tolist())

    # it does not produce an official mAP score (but the difference is negligible)
    mAP = average_precision_score(target_total, soft_total) * 100    
    return mAP


if __name__ == '__main__':
    main()
