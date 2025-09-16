#!/usr/bin/env bash
#liguanjun@2024.06
stage=$1

# Configure the path to your local AVA dataset directory
AVADataPath=/path/to/your/AVA

mkdir -p ./predata


# dowload the AVA dataset
if [ $stage -eq 0 ]; then
	python train.py --dataPathAVA $AVADataPath --download
fi

# get some csv files for processing
if [ $stage -eq 1 ]; then
    python ./local/get_csv_split.py --dataPathAVA $AVADataPath --out_path ./predata/csv_split
fi

# extract the F_bg in Eq.(8)
if [ $stage -eq 2 ]; then 
    python ./local/prepare_background.py --AVADataPath=$AVADataPath --out_path=$AVADataPath/clips_videos_background
    mkdir -p ./predata/background_process/${name}_list
    for name in train val; do
        echo deal with ${name} list
        python ./local/get_leaf_dir_list.py --AVADataPath=$AVADataPath/clips_videos/$name --out_file=./predata/background_process/${name}_list
    done
fi

# resize the background to 224*224
if [ $stage -eq 3 ]; then
    background_H=224
    nj=40
    mkdir -p ./predata/background_process/split
    for name in train val; do
        file_split_scp=""
        for job in $(seq $nj); do
            file_split_scp="$file_split_scp ./predata/background_process/split/${name}_file_list.$job"
        done
        ./local/utils/split_scp.pl ./predata/background_process/${name}_list $file_split_scp
        mkdir -p ./log/resize_background
        mkdir -p $AVADataPath/clips_videos_background_resize
        ./local/utils/run.pl JOB=1:$nj \
        ./log/resize_background/${set}/resize_background.JOB.log \
            python ./local/resize_background.py --input_file_list=./predata/background_process/split/${name}_file_list.JOB \
                                            --AVADataPath $AVADataPath \
                                            --out_path=$AVADataPath/clips_videos_background_resize_${background_H} \
                                            --size=$background_H \
    
    done
fi

# extract the F_body in Eq.(6)
if [ $stage -eq 4 ]; then
    nj=40
    mkdir -p ./predata/facebody_process/split
    for name in train val; do
        if [[ "$name" == "val" ]];then
            nj=33
        fi
        echo deal with $name 
        find $(pwd)/predata/csv_split/$name -type f > ./predata/facebody_process/${name}_list
        file_split_scp=""
        for job in $(seq $nj); do
            file_split_scp="$file_split_scp ./predata/facebody_process/split/${name}_file_list.$job"
        done
        ./local/utils/split_scp.pl ./predata/facebody_process/${name}_list $file_split_scp
        mkdir -p ./log/facebody_process
        mkdir -p $AVADataPath/clips_videos_face_body
        ./local/utils/run.pl JOB=1:$nj \
        ./log/facebody_process/${set}/facebody_process.JOB.log \
            python ./local/prepare_facebody.py --input_file_list=./predata/facebody_process/split/${name}_file_list.JOB \
                                               --AVADataPath $AVADataPath \
                                               --out_path=$AVADataPath/clips_videos_face_body \
    
    done

    # python ./local/resize_background.py --AVADataPath=$AVADataPath --out_path=$AVADataPath/clips_videos_background_resize --size=$background_H
fi

# extract the F_head in Eq.(3)
if [ $stage -eq 5 ]; then
    nj=40
    mkdir -p ./predata/face_process/split
    for name in train val; do
        if [[ "$name" == "val" ]];then
            nj=33
        fi
        echo deal with $name 
        find $(pwd)/predata/csv_split/$name -type f > ./predata/face_process/${name}_list
        file_split_scp=""
        for job in $(seq $nj); do
            file_split_scp="$file_split_scp ./predata/face_process/split/${name}_file_list.$job"
        done
        ./local/utils/split_scp.pl ./predata/face_process/${name}_list $file_split_scp
        mkdir -p ./log/face_process
        mkdir -p $AVADataPath/clips_videos_face_large_region
        ./local/utils/run.pl JOB=1:$nj \
        ./log/face_process/${set}/face_process.JOB.log \
            python ./local/prepare_largeface.py --input_file_list=./predata/face_process/split/${name}_file_list.JOB \
                                               --AVADataPath $AVADataPath \
                                               --out_path=$AVADataPath/clips_videos_face_large_region \
    
    done

    # python ./local/resize_background.py --AVADataPath=$AVADataPath --out_path=$AVADataPath/clips_videos_background_resize --size=$background_H
fi

# extract the F_smallfae in Eq.(4)
if [ $stage -eq 6 ]; then
    echo extract the small region face
    nj=40
    mkdir -p ./predata/face_process/split
    for name in train val; do
        if [[ "$name" == "val" ]];then
            nj=33
        fi
        echo deal with $name 
        find $(pwd)/predata/csv_split/$name -type f > ./predata/face_process/${name}_list
        file_split_scp=""
        for job in $(seq $nj); do
            file_split_scp="$file_split_scp ./predata/face_process/split/${name}_file_list.$job"
        done
        ./local/utils/split_scp.pl ./predata/face_process/${name}_list $file_split_scp
        mkdir -p ./log/face_process
        mkdir -p $AVADataPath/clips_videos_face_small_region
        ./local/utils/run.pl JOB=1:$nj \
        ./log/face_process/${set}/face_process.JOB.log \
            python ./local/prepare_smallface.py --input_file_list=./predata/face_process/split/${name}_file_list.JOB \
                                               --AVADataPath $AVADataPath \
                                               --out_path=$AVADataPath/clips_videos_face_small_region \
    
    done

    # python ./local/resize_background.py --AVADataPath=$AVADataPath --out_path=$AVADataPath/clips_videos_background_resize --size=$background_H
fi

# extract the F_mouse in Eq.(5)
if [ $stage -eq 7 ]; then
    mkdir ./predata/face_process/${name}_list
    for name in train val; do
        echo deal with ${name} list
        python ./local/get_leaf_dir_list.py --AVADataPath=$AVADataPath/clips_videos/$name --out_file=./predata/face_process/${name}_list
    done
    
    nj=64
    mkdir -p ./predata/face_process/split
    for name in train val; do
        echo deal with $name
        file_split_scp=""
        for job in $(seq $nj); do
            file_split_scp="$file_split_scp ./predata/face_process/split/${name}_file_list.$job"
        done
        ./local/utils/split_scp.pl ./predata/face_process/${name}_list   $file_split_scp
        mkdir -p ./log/extract_mouse
        mkdir -p $AVADataPath/clips_videos_down_face
        ./local/utils/run.pl JOB=1:$nj \
        ./log/extract_mouse/${set}/extract_mouse.JOB.log \
            python ./local/prepare_halfface.py --input_file_list=./predata/face_process/split/${name}_file_list.JOB \
                                                        --AVADataPath $AVADataPath \
                                                        --down_out_path=$AVADataPath/clips_videos_down_face \
                                                        --csv_split_root=./predata/csv_split/$name
    
    done
fi

# extract the F_largebody in Eq.(7)
if [ $stage -eq 8 ]; then
    nj=40
    mkdir -p ./predata/facebody_large_process/split
    for name in train val; do
        if [[ "$name" == "val" ]];then
            nj=33
        fi
        echo deal with $name 
        find $(pwd)/predata/csv_split/$name -type f > ./predata/facebody_large_process/${name}_list
        file_split_scp=""
        for job in $(seq $nj); do
            file_split_scp="$file_split_scp ./predata/facebody_large_process/split/${name}_file_list.$job"
        done
        ./local/utils/split_scp.pl ./predata/facebody_large_process/${name}_list $file_split_scp
        mkdir -p ./log/facebody_large_process
        mkdir -p $AVADataPath/clips_videos_face_body
        ./local/utils/run.pl JOB=1:$nj \
        ./log/facebody_large_process/${set}/facebody_large_process.JOB.log \
            python ./local/prepare_facebody_large.py --input_file_list=./predata/facebody_large_process/split/${name}_file_list.JOB \
                                               --AVADataPath $AVADataPath \
                                               --out_path=$AVADataPath/clips_videos_face_body_large \
    
    done
fi