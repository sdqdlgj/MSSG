#!/usr/bin/env bash
#liguanjun@2024.06
stage=$1

# Configure the path to your local AVA dataset directory
AVADataPath=/path/to/your/AVA

mkdir -p ./predata
mkdir -p ./predata/clip_info


# Light ASD pretraining
# Please put the best checkpoint into ./predata/pretrain_model
# You can place multiple high-quality checkpoints into the ./predata/pretrain_model/xx_model folder, 
# for example: ./predata/pretrain_model/face_model.
# Except for the background model, we found that averaging multiple checkpoints can achieve better performance.
if [ $stage -eq 0 ]; then
    for train_type in face, background, face_body, face_large, face_down, face_body_large, face_body; do
        encoder_struct='32,64,128'
        small_training=0
        small_trainset=293
        small_valset=800
        batchSize=1000    # should more than 1000
        GPU=0
        maxEpoch=60
        nj=20
        background_H=224
        savePath=exps/small_exp_$(date +"%Y-%m-%d-%H:%M:%S")
        if [ $small_training -eq 0 ]; then
            savePath=exps/exp_$(date +"%Y-%m-%d-%H:%M:%S")
        fi

        CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=${GPU} python train.py --dataPathAVA $AVADataPath \
                                                    --savePath $savePath \
                                                    --small_training $small_training \
                                                    --small_trainset $small_trainset \
                                                    --small_valset $small_valset \
                                                    --batchSize $batchSize \
                                                    --maxEpoch $maxEpoch \
                                                    --nDataLoaderThread $nj \
                                                    --train_type $train_type \
                                                    --background_H $background_H \
                                                    --encoder_struct $encoder_struct
    done
 
fi

# initialize node e_face in Eq.(12)
if [ $stage -eq 1 ]; then
    GPU=1
    python ./local/get_clip_info.py --dataPathAVA $AVADataPath --out_path ./predata/clip_info
    CUDA_VISIBLE_DEVICES=${GPU} python ./local/extract_face_feats_average.py --dataPathAVA $AVADataPath \
                                   --clip_info_path ./predata/clip_info \
                                   --face_model_path /share/home/liguanjun/src/LGJ_PretrainASD/predata/pretrain_model/face_model \
                                   --out_path ./predata/face_feats_info \

fi

# initialize node e_bg in Eq.(12)
if [ $stage -eq 2 ]; then
    GPU=0
    CUDA_VISIBLE_DEVICES=${GPU} python ./local/extract_background_feats.py --dataPathAVA $AVADataPath \
                                   --clip_info_path ./predata/clip_info \
                                   --background_model ./predata/pretrain_model/background_model.model \
                                   --out_path ./predata/background_feats_info \
                                   --background_H 224


fi

# initialize node e_smallface in Eq.(12)
if [ $stage -eq 3 ]; then
    GPU=1
    CUDA_VISIBLE_DEVICES=${GPU} python ./local/extract_face_small_feats_average.py --dataPathAVA $AVADataPath \
                            --clip_info_path ./predata/clip_info \
                            --face_model_path ./predata/pretrain_model/face_small_model \
                            --out_path ./predata/face_small_feats_info \


fi

# initialize node e_body in Eq.(12)
if [ $stage -eq 4 ]; then 
    GPU=1
    CUDA_VISIBLE_DEVICES=${GPU} python ./local/extract_face_body_feats_average.py --dataPathAVA $AVADataPath \
                        --clip_info_path ./predata/clip_info \
                        --face_model_path ./predata/pretrain_model/body_model \
                        --out_path ./predata/face.body_feats_info \

fi


# initialize node e_head in Eq.(12)
if [ $stage -eq 5 ]; then
    GPU=0
    CUDA_VISIBLE_DEVICES=${GPU} python ./local/extract_face_large_feats_average.py --dataPathAVA $AVADataPath \
                                --clip_info_path ./predata/clip_info \
                                --face_model_path ./predata/pretrain_model/face_large_model \
                                --out_path ./predata/face_large_feats_info \

fi


# initialize node e_mouse in Eq.(12)
if [ $stage -eq 6 ]; then
    GPU=1
    CUDA_VISIBLE_DEVICES=${GPU} python ./local/extract_face_down_feats_average.py --dataPathAVA $AVADataPath \
                            --clip_info_path ./predata/clip_info \
                            --face_model_path ./predata/pretrain_model/face_down_model \
                            --out_path ./predata/face_down_feats_info \

fi

# initialize node e_largebody in Eq.(12)
if [ $stage -eq 7 ]; then
    GPU=0
    CUDA_VISIBLE_DEVICES=${GPU} python ./local/extract_face_body_large_feats_average.py --dataPathAVA $AVADataPath \
                            --clip_info_path ./predata/clip_info \
                            --face_model_path ./predata/pretrain_model/body_large_model \
                            --out_path ./predata/face_body_large_feats_info \
                            
fi

# generate speaker graph G
if [ $stage -eq 8 ]; then
    mkdir -p ./predata/video_size_info
    python ./local/get_size_of_each_video.py --AVADataPath $AVADataPath --out_path=./predata/video_size_info
    nj=33
    numv=2000
    time_edge=0
    for set in train val; do
        if [[ "$name" == "val" ]];then
            nj=33
        fi
        echo deal with $set using $nj workers ...
        csv_split_path=./predata/csv_split/${set}
        face_feats_info_path=./predata/face_feats_info/${set}
        background_feats_info_path=./predata/background_feats_info/${set}
        face_body_feats_info_path=./predata/face.body_feats_info/${set}
        face_large_feats_info_path=./predata/face_large_feats_info/${set}
        face_small_feats_info_path=./predata/face_small_feats_info/${set}
        face_down_feats_info_path=./predata/face_down_feats_info/${set}
        face_body_large_feats_info_path=./predata/face_body_large_feats_info/${set}
        out_path=./graphs/graph_${numv}_${time_edge}/${set}/processed
        mkdir -p $out_path
        # split read files for multi-processing
        mkdir -p ./log/generate_graph
        find $csv_split_path -type f > ./log/generate_graph/${set}_file_list
        file_split_scp=""
        for job in $(seq $nj); do
            file_split_scp="$file_split_scp ./log/generate_graph/${set}_file_list.$job"
        done
        ./local/utils/split_scp.pl ./log/generate_graph/${set}_file_list $file_split_scp
        # begin multi-processing
        ./local/utils/run.pl JOB=1:$nj \
                        ./log/generate_graph/${set}/generate_graph.JOB.log \
                        python ./local/generate_graph.py --input_file_list=./log/generate_graph/${set}_file_list.JOB \
                                                        --face_feats_info_path=$face_feats_info_path \
                                                        --background_feats_info_path=$background_feats_info_path \
                                                        --face_body_feats_info_path=$face_body_feats_info_path \
                                                        --face_large_feats_info_path=$face_large_feats_info_path \
                                                        --face_small_feats_info_path=$face_small_feats_info_path \
                                                        --face_down_feats_info_path=$face_down_feats_info_path \
                                                        --face_body_large_feats_info_path=$face_body_large_feats_info_path \
                                                        --out_path=$out_path \
                                                        --numv=$numv \
                                                        --time_edge=$time_edge \
                                                        --video_size_info=./predata/video_size_info/size_dic.pkl


    done
fi