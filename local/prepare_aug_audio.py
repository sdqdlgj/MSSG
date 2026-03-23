import os, subprocess, glob, pandas, tqdm, cv2, numpy
from scipy.io import wavfile
import argparse
import torch
import random
import numpy as np 

def run(args):
    random.seed(111)
    np.random.seed(111)
    iii =  0
    for name in ['train']:
        clip_info = torch.load(os.path.join(args.clip_info_path, name+'_clip_info.pt'))
        audio_set = {}
        pre_video_name = 'xxx'
        for audio_clip_name in clip_info.keys():
            video_name = audio_clip_name[:11]
            ori_audio_path = os.path.join(args.dataPathAVA, 'clips_audios', name, video_name, audio_clip_name+".wav")
            _, cur_audio = wavfile.read(ori_audio_path)
            if video_name != pre_video_name:
                if pre_video_name != 'xxx':
                    audio_set[pre_video_name] = audio
                    print(f'loaded {pre_video_name} ...')
                    # iii += 1
                    # if iii == 2:
                    #     break
                pre_video_name = video_name
                audio = cur_audio                
            else:
                audio = np.concatenate((audio,cur_audio), axis=0)
        audio_set[video_name] = audio
        pre_video_name = 'xxx'
        for audio_clip_name in clip_info.keys():
            if video_name != pre_video_name:
                print(f'aug {video_name}')
                pre_video_name = video_name  
            video_name = audio_clip_name[:11]
            if not os.path.exists(os.path.join(args.out_path, name, video_name)):
                os.makedirs(os.path.join(args.out_path, name, video_name))
            ori_audio_path = os.path.join(args.dataPathAVA, 'clips_audios', name, video_name, audio_clip_name+".wav")
            _, cur_audio = wavfile.read(ori_audio_path)
            noiseName =  list(audio_set.keys())
            noiseName.remove(video_name)
            noiseName = random.choice(noiseName)
            noise = audio_set[noiseName]
            snr = [random.uniform(-5, 5)]
            audio_size = cur_audio.shape[0]
            start = random.randint(0, noise.shape[0] - audio_size - 2)
            end = start + audio_size
            noise = noise[start:end]
            noiseDB = 10 * numpy.log10(numpy.mean(abs(noise ** 2)) + 1e-4)
            cleanDB = 10 * numpy.log10(numpy.mean(abs(cur_audio ** 2)) + 1e-4)
            noise = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noise
            cur_audio = cur_audio + noise
            cur_audio = cur_audio.astype(numpy.int16)
            wavfile.write(os.path.join(args.out_path, name, video_name, audio_clip_name+".wav"), 16000, cur_audio)    

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument('--dataPathAVA', type=str, dest='dataPathAVA', default='/share/home/liguanjun/data/AVA', help="dataPathAVA")
    parser.add_argument('--out_path', type=str, dest='out_path', default='/share/home/liguanjun/data/AVA/clips_audios_aug', help="out_path")
    parser.add_argument('--clip_info_path', type=str, dest='clip_info_path', default='./predata/clip_info', help="clip_info_path")
    args = parser.parse_args()
    run(args)
