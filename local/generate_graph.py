import os
import argparse
import pickle
import sys
import csv
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from torch_geometric.data.makedirs import makedirs

def decode_feature(feature_data):
    feature_data = feature_data[1:-1]
    feature_data = feature_data.split(',')
    return np.array([float(fd) for fd in feature_data], dtype=np.float32)

def run(args):
    ct=1
    # get input list
    with open(args.input_file_list, 'r') as file:
        files = [line.strip() for line in file.readlines()]

    id_dict = {}
    vstamp_dict = {}
    id_ct = 0
    ustamp = 0
    
    ## iterating over videos(features) in training/validation set
    for fl in files:
        print(f'deal with {fl}')
        file_name = os.path.basename(fl)
        ## load the current feature csv file
        with open(fl, newline='') as f:
            reader = csv.reader(f)
            data_f = list(reader)
        
        with open(args.video_size_info, 'rb') as fid:
            size_dic = pickle.load(fid)
              
        with open(os.path.join(args.face_feats_info_path, file_name)) as f:
            reader = csv.reader(f)
            face_feat_data = list(reader)
        face_feat_info = {}
        for item in face_feat_data:
            # face_feat_info[item[0]] = np.concatenate((decode_feature(item[1]),decode_feature(item[2]),decode_feature(item[3])), axis=0)
            face_feat_info[item[0]] = decode_feature(item[1])
            
        with open(os.path.join(args.background_feats_info_path, file_name)) as f:
            reader = csv.reader(f)
            background_feat_data = list(reader)
        background_feat_info = {}
        for item in background_feat_data:
            background_feat_info[item[0]] = decode_feature(item[1])
            

        with open(os.path.join(args.face_body_feats_info_path, file_name)) as f:
            reader = csv.reader(f)
            face_body_feat_data = list(reader)
        face_body_feat_info = {}
        for item in face_body_feat_data:
            face_body_feat_info[item[0]] = decode_feature(item[1])
            
        with open(os.path.join(args.face_large_feats_info_path, file_name)) as f:
            reader = csv.reader(f)
            face_large_feats_data = list(reader)
        face_large_feat_info = {}
        for item in face_large_feats_data:
            face_large_feat_info[item[0]] = decode_feature(item[1])
        
        with open(os.path.join(args.face_small_feats_info_path, file_name)) as f:
            reader = csv.reader(f)
            face_small_feats_data = list(reader)
        face_small_feat_info = {}
        for item in face_small_feats_data:
            face_small_feat_info[item[0]] = decode_feature(item[1])

        with open(os.path.join(args.face_down_feats_info_path, file_name)) as f:
            reader = csv.reader(f)
            face_down_feats_data = list(reader)
        face_down_feat_info = {}
        for item in face_down_feats_data:
            face_down_feat_info[item[0]] = decode_feature(item[1])     
        
        with open(os.path.join(args.face_body_large_feats_info_path, file_name)) as f:
            reader = csv.reader(f)
            face_body_large_feats_data = list(reader)
        face_body_large_feat_info = {}
        for item in face_body_large_feats_data:
            face_body_large_feat_info[item[0]] = decode_feature(item[1])          
        

        # we sort the rows by their time-stamps
        data_f.sort(key = lambda x: float(x[1]))

        num_v = args.numv
        count_gp = 1
        len_data = len(data_f)
        # iterating over blocks of face-boxes(or the rows) of the current feature file
        for i in range(0, len_data, args.numv):

            ## in pygeometric edges are stored in source-target/directed format ,i.e, for us (source_vertices[i], source_vertices[i]) is an edge for all i
            source_vertices = []
            target_vertices = []

            # x is the list to store the vertex features ; x[i,:] is the feature of the i-th vertex
            x = []
            # y is the list to store the vertex labels ; y[i] is the label of the i-th vertex
            y = []
            # identity and times are two lists keep track of idenity and time stamp of the current vertex
            identity = []
            times = []

            unique_id = []
            identity2times = {}
            times2identity = {}
            ##------------------------------
            ## this block computes the index of the start facebox and the last
            if i+num_v <= len_data:
                start_g = i
                end_g = i+num_v
            else:
                start_g = i
                end_g = len_data
            ##--------------------------------------

            ### we go over the face-boxes of the current partition and construct their edges, collect their features within this for loop
            for j in range(start_g, end_g):
                stamp_marker = data_f[j][1] + data_f[j][0]
                id_marker = data_f[j][2] + str(ct)

                if stamp_marker not in vstamp_dict:
                    vstamp_dict[stamp_marker] = ustamp
                    ustamp = ustamp + 1

                if id_marker  not in id_dict:
                    id_dict[id_marker] = id_ct
                    id_ct = id_ct + 1
                #---------------------------------------------

                vte = (data_f[j][0], float(data_f[j][1]), data_f[j][2])

                ## parse the current facebox's feature from data_f
                face_feat = face_feat_info[data_f[j][7]+":"+data_f[j][1]]
                background_feat = background_feat_info[data_f[j][7]+":"+data_f[j][1]]
                face_body_feat = face_body_feat_info[data_f[j][7]+":"+data_f[j][1]]
                face_large_feat = face_large_feat_info[data_f[j][7]+":"+data_f[j][1]]
                face_small_feat = face_small_feat_info[data_f[j][7]+":"+data_f[j][1]]
                face_down_feat = face_down_feat_info[data_f[j][7]+":"+data_f[j][1]]
                face_body_large = face_body_large_feat_info[data_f[j][7]+":"+data_f[j][1]]
                feat = np.concatenate((face_feat, 
                                       face_body_feat, 
                                       face_large_feat, 
                                       background_feat, 
                                       face_small_feat, 
                                       face_down_feat,
                                       face_body_large,
                                       ), axis=0)
                feat = np.expand_dims(feat, axis=0)
                x.append(feat)
                label = float(data_f[j][8])
                if label == 2.0:
                    label = 0.0
                y.append(label)

                ## append time and identity of i-th vertex to the list of time stamps and identitites
                times.append(float(data_f[j][1]))
                identity.append(data_f[j][7])

                cur_identity = data_f[j][7]
                cur_times = float(data_f[j][1])
                if cur_identity not in identity2times.keys():
                    identity2times[cur_identity] = [cur_times]
                else:
                    identity2times[cur_identity].append(cur_times)
                    
                if cur_times not in times2identity.keys():
                    times2identity[cur_times] = [cur_identity]
                else:
                    times2identity[cur_times].append(cur_identity)
            
            x_i = -1
            for j in range(start_g, end_g):
                x_i = x_i + 1
                cur_times = float(data_f[j][1])
                cur_identity = data_f[j][7]
                x1, y1, x2, y2 = float(data_f[j][2]), float(data_f[j][3]), float(data_f[j][4]), float(data_f[j][5])
                vte_spe = [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]
                vte_spe = np.array(vte_spe).reshape(1, 4)
                video_name = data_f[j][0]
                video_size = size_dic[video_name]
                face_size = np.array(float(video_size[0]) * (x2-x1)).reshape(1, 1)
                speaker_num = len(times2identity[cur_times])
                speaker_num = np.array(speaker_num).reshape(1, 1)
                x[x_i] = np.concatenate((x[x_i], face_size), axis=1)
                x[x_i] = np.concatenate((x[x_i], vte_spe), axis=1)
                x[x_i] = np.concatenate((x[x_i], speaker_num), axis=1)
                
            
            edge_attr = []
            num_edge = 0

            speaker_list = list(set(identity))
            speaker_num = len(speaker_list)
            ## iterating over pairs of vertices of the current partition and assign edges accodring to some criterion
            link_flag = np.zeros((speaker_num,speaker_num))
            for j in range(0, end_g - start_g):
                for k in range(0, end_g - start_g):

                    id_cond = identity[j]==identity[k]
                    j_id_idx = speaker_list.index(identity[j])
                    k_id_idx = speaker_list.index(identity[k])
                    link_cond = link_flag[j_id_idx,k_id_idx]==1
                    
                    # time difference between j-th and k-th vertex
                    time_gap = times[j]-times[k]
                    
                    # link_cond = False
                    # if time_gap!=0 and not id_cond:
                    #     time_list_k = identity2times[identity[k]]
                    #     if times[j] not in time_list_k:
                    #         time_list_j = identity2times[identity[j]]
                    #         intersection = set(time_list_j) & set(time_list_k)
                    #         if len(intersection) != 0:                                
                    #             inter_np = np.array(list(intersection))
                    #             distances = np.abs(inter_np - times[j])
                    #             min_index = np.argmin(distances)
                    #             closest_element = inter_np[min_index]
                    #             if times[k] == closest_element:
                    #                  link_cond = True
 
                    # self connection
                    if abs(time_gap) <= 0.0 and id_cond:
                        source_vertices.append(j)
                        target_vertices.append(k)
                        num_edge = num_edge + 1
                        edge_attr.append(111)
                    
                    # different face in the same frame
                    if abs(time_gap) <= 0.0 and not id_cond:
                        source_vertices.append(j)
                        target_vertices.append(k)
                        num_edge = num_edge + 1
                        edge_attr.append(0)

            ##--------------- convert vertex features,edges,edge_features, labels to tensors
            x = torch.FloatTensor(np.concatenate(x, axis=0))
            edge_index = torch.LongTensor([source_vertices, target_vertices])
            edge_attr = torch.FloatTensor(edge_attr)
            y = torch.FloatTensor(y).unsqueeze(1)
            #----------------

            ## creates the graph data object that stores (features,edges,labels)
            data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

            ### save the graph data file with appropriate name; They are named as follows: videoname_1.pt,video_name_2.pt and so on
            torch.save(data, os.path.join(args.out_path,data_f[0][0])+ '_{:03d}.pt'.format(count_gp))
            count_gp = count_gp + 1    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--input_file_list', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/log/generate_graph/train_file_list.2', help="input_file_list")
    parser.add_argument('--face_feats_info_path', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/face_feats_info/train', help="face_feats_info_path")
    parser.add_argument('--background_feats_info_path', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/background_feats_info/train', help="background_feats_info_path")
    parser.add_argument('--face_body_feats_info_path', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/face.body_feats_info/train', help="face_body_feats_info_path")
    parser.add_argument('--face_large_feats_info_path', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/face_large_feats_info/train', help="face_large_feats_info_path")
    parser.add_argument('--face_small_feats_info_path', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/face_small_feats_info/train', help="face_small_feats_info_path")
    parser.add_argument('--face_down_feats_info_path', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/face_down_feats_info/train', help="face_down_feats_info_path")   
    parser.add_argument('--face_body_large_feats_info_path', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/face_body_large_feats_info/train', help="face_body_large_feats_info_path")   
    parser.add_argument('--out_path', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/graphs/test', help="out_path")
    parser.add_argument('--numv', type=int, default=2000, help='number of nodes')
    parser.add_argument('--time_edge', type=float, default=0.9, help='time threshold')
    parser.add_argument('--video_size_info', type=str, default='/share/home/liguanjun/src/LGJ_PretrainASD/predata/video_size_info/size_dic.pkl', help='video_size_info')
    args = parser.parse_args()
    run(args)