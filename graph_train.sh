#!/usr/bin/env bash
#liguanjun@2024.06

GPU=0
numv=2000
batch_size=64
lr=0.001
graph_path=./graphs
python train_graph.py --gpu_id=$GPU \
                      --numv=$numv \
                      --batch_size=$batch_size \
                      --graph_path=$graph_path \
                      --lr=$lr