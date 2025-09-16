import os, sys, torch, numpy, cv2, random, glob, python_speech_features
from scipy.io import wavfile
from torchvision.transforms import RandomCrop

from torch_geometric.data import Data
from torch_geometric.data import Dataset
import numpy as np

def generate_audio_set(dataPath, batchList):
    audioSet = {}
    for line in batchList:
        data = line.split('\t')
        videoName = data[0][:11]
        dataName = data[0]
        _, audio = wavfile.read(os.path.join(dataPath, videoName, dataName + '.wav'))
        audioSet[dataName] = audio
    return audioSet

def overlap(dataName, audio, audioSet):   
    noiseName =  list(audioSet.keys())
    noiseName.remove(dataName)
    noiseName = random.choice(noiseName)
    
    noiseAudio = audioSet[noiseName]    
    snr = [random.uniform(-5, 5)]
    if len(noiseAudio) < len(audio):
        shortage = len(audio) - len(noiseAudio)
        noiseAudio = numpy.pad(noiseAudio, (0, shortage), 'wrap')
    else:
        noiseAudio = noiseAudio[:len(audio)]
    noiseDB = 10 * numpy.log10(numpy.mean(abs(noiseAudio ** 2)) + 1e-4)
    cleanDB = 10 * numpy.log10(numpy.mean(abs(audio ** 2)) + 1e-4)
    noiseAudio = numpy.sqrt(10 ** ((cleanDB - noiseDB - snr) / 10)) * noiseAudio
    audio = audio + noiseAudio    
    return audio.astype(numpy.int16)

# def load_audio(data, dataPath, numFrames, audioAug, audioSet = None):
#     dataName = data[0]
#     fps = float(data[2])    
#     audio = audioSet[dataName]    
#     if audioAug == True:
#         augType = random.randint(0,1)
#         if augType == 1:
#             audio = overlap(dataName, audio, audioSet)
#         else:
#             audio = audio
#     min_zero_elements = int(audio.shape[0] * 0.01)
#     max_zero_elements = int(audio.shape[0] * 0.03)
#     zero_length = numpy.random.randint(min_zero_elements, max_zero_elements + 1)
#     start_index = numpy.random.randint(0, audio.shape[0] - zero_length + 1)
#     mask_flag = random.randint(0,1)
#     if mask_flag:
#         audio_mask = audio.copy()
#         audio_mask[start_index:start_index + zero_length] = 0
#         audio = audio_mask
#     # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
#     audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
#     maxAudio = int(numFrames * 4)
#     if audio.shape[0] < maxAudio:
#         shortage    = maxAudio - audio.shape[0]
#         audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
#     audio = audio[:int(round(numFrames * 4)),:]  
#     return audio


# def load_visual(data, dataPath, numFrames, H, visualAug): 
#     dataName = data[0]
#     videoName = data[0][:11]
#     faceFolderPath = os.path.join(dataPath, videoName, dataName)
#     faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
#     sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
#     faces = []
#     H = H
#     if visualAug == True:
#         M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
#         augType = random.choice(['orig', 'flip', 'rotate']) 
#     else:
#         augType = 'orig'
#     for faceFile in sortedFaceFiles[:numFrames]:
#         face = cv2.imread(faceFile)
#         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#         face = cv2.resize(face, (H,H))
#         if augType == 'orig':
#             face = face
#         elif augType == 'flip':
#             face = cv2.flip(face, 1)
#         elif augType == 'rotate':
#             face = cv2.warpAffine(face, M, (H,H))
        
#         augType_t = random.choice(['orig', 'orig', 'orig', 'crop']) 
#         if augType_t == 'crop':
#             new = int(H*random.uniform(0.7, 1))
#             x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
#             face = cv2.resize(face[y:y+new, x:x+new] , (H,H))
        
#         faces.append(face) 
        
#     faces = numpy.array(faces)
#     return faces

def load_audio(data, dataPath, numFrames, audioAug, audioSet = None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    
    if audioAug == True:
        augType = random.randint(0,1)
        if augType == 1:
            audio = overlap(dataName, audio, audioSet)
        else:
            audio = audio
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    return audio

def load_visual(data, dataPath, numFrames, H, visualAug): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = H
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
    faces = numpy.array(faces)
    return faces

def load_visual_graph_train(data, dataPath, numFrames): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = 112
    new = int(H*random.uniform(0.3, 0.7))
    x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
    kernel_size = numpy.random.randint(5, 15)
    M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
    augType = random.choice(['orig', 'flip', 'crop', 'rotate'])
    time_mask = 0 
    min_zero_elements = int(len(sortedFaceFiles[:numFrames]) * 0.01)
    max_zero_elements = int(len(sortedFaceFiles[:numFrames]) * 0.03)
    zero_length = numpy.random.randint(min_zero_elements, max_zero_elements + 1)
    start = numpy.random.randint(0, len(sortedFaceFiles[:numFrames]) - zero_length + 1)
    end = start + zero_length
    count_num = 0
    for faceFile in sortedFaceFiles[:numFrames]:
        if time_mask and start<count_num<end:
            faces.append(numpy.zeros((112, 112), dtype=numpy.uint8))
        else:
            face = cv2.imread(faceFile)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (H,H))
            if augType == 'orig':
                faces.append(face)
            elif augType == 'flip':
                faces.append(cv2.flip(face, 1))
            elif augType == 'crop':
                faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
            elif augType == 'rotate':
                faces.append(cv2.warpAffine(face, M, (H,H)))
        count_num += 1
    faces = numpy.array(faces)
    return faces



def load_audio_graph_train(data, dataPath, numFrames, audioSet = None):
    dataName = data[0]
    fps = float(data[2])    
    audio = audioSet[dataName]    
    augType = random.randint(0,1)
    min_zero_elements = int(audio.shape[0] * 0.01)
    max_zero_elements = int(audio.shape[0] * 0.03)
    zero_length = numpy.random.randint(min_zero_elements, max_zero_elements + 1)
    start_index = numpy.random.randint(0, audio.shape[0] - zero_length + 1)
    mask_flag = 0
    if mask_flag:
        audio_mask = audio.copy()
        audio_mask[start_index:start_index + zero_length] = 0
        audio = audio_mask
    # fps is not always 25, in order to align the visual, we modify the window and step in MFCC extraction process based on fps
    audio = python_speech_features.mfcc(audio, 16000, numcep = 13, winlen = 0.025 * 25 / fps, winstep = 0.010 * 25 / fps)
    
    maxAudio = int(numFrames * 4)
    if audio.shape[0] < maxAudio:
        shortage    = maxAudio - audio.shape[0]
        audio     = numpy.pad(audio, ((0, shortage), (0,0)), 'wrap')
    audio = audio[:int(round(numFrames * 4)),:]  
    return audio


def load_visual_face(data, dataPath, numFrames, H, visualAug): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    faces = []
    H = H
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            faces.append(face)
        elif augType == 'flip':
            faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            faces.append(cv2.warpAffine(face, M, (H,H)))
        break
    faces = numpy.array(faces)
    return faces

def load_visual_multiscale(data, scale_num, dataPath, numFrames, H, visualAug): 
    dataName = data[0]
    videoName = data[0][:11]
    faceFolderPath = os.path.join(dataPath, videoName, dataName)
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    H = H
    # scale_num = 11
    delta_x1 = np.linspace(0, 0.25, scale_num)
    delta_x2 = np.linspace(0, 0.25, scale_num)
    delta_y1 = np.linspace(0, 0.25, scale_num)
    delta_y2 = np.linspace(0, 0.25, scale_num)
    if visualAug == True:
        new = int(H*random.uniform(0.7, 1))
        x, y = numpy.random.randint(0, H - new), numpy.random.randint(0, H - new)
        M = cv2.getRotationMatrix2D((H/2,H/2), random.uniform(-15, 15), 1)
        # TODO: Whether to use crop?
        augType = random.choice(['orig', 'flip', 'crop', 'rotate']) 
    else:
        augType = 'orig'
    faces = []
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        y_len, x_len = face.shape
        multiscale_faces = []
        for sample_i in range(delta_x1.shape[0]):
            new_x1 = int(x_len * delta_x1[sample_i])
            new_x2 = int(x_len - x_len * delta_x2[sample_i])
            new_y1 = int(y_len * delta_y1[sample_i])
            new_y2 = int(y_len - y_len * delta_y2[sample_i])
            cur_face = face[new_y1:new_y2, new_x1:new_x2]
            cur_face = cv2.resize(cur_face, (H,H))
            if augType == 'orig':
                multiscale_faces.append(cur_face)
            elif augType == 'flip':
                multiscale_faces.append(cv2.flip(cur_face, 1))
            elif augType == 'crop':
                multiscale_faces.append(cv2.resize(cur_face[y:y+new, x:x+new] , (H,H))) 
            elif augType == 'rotate':
                multiscale_faces.append(cv2.warpAffine(cur_face, M, (H,H)))
        multiscale_faces = numpy.array(multiscale_faces)
        faces.append(multiscale_faces)
    faces = numpy.array(faces)
    ori_faces = []
    faceFolderPath = faceFolderPath.replace("_face_large_region", "")
    faceFiles = glob.glob("%s/*.jpg"%faceFolderPath)
    sortedFaceFiles = sorted(faceFiles, key=lambda data: (float(data.split('/')[-1][:-4])), reverse=False) 
    for faceFile in sortedFaceFiles[:numFrames]:
        face = cv2.imread(faceFile)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, (H,H))
        if augType == 'orig':
            ori_faces.append(face)
        elif augType == 'flip':
            ori_faces.append(cv2.flip(face, 1))
        elif augType == 'crop':
            ori_faces.append(cv2.resize(face[y:y+new, x:x+new] , (H,H))) 
        elif augType == 'rotate':
            ori_faces.append(cv2.warpAffine(face, M, (H,H)))
    ori_faces = numpy.array(ori_faces)
    ori_faces = np.expand_dims(ori_faces, axis=1)
    faces = np.concatenate((ori_faces, faces), axis=1)    
    faces = np.transpose(faces, (1, 0, 2, 3))
    return faces


def load_label(data, numFrames):
    res = []
    labels = data[3].replace('[', '').replace(']', '')
    labels = labels.split(',')
    for label in labels:
        res.append(int(label))
    res = numpy.array(res[:numFrames])
    return res

class train_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, H, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []
        self.H = H      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)               
        start = 0       
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end     

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')            
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))  
            visualFeatures.append(load_visual(data, self.visualPath,numFrames, self.H, visualAug = True))
            labels.append(load_label(data, numFrames))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)


class val_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, H, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.H = H  
    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual(data, self.visualPath,numFrames, self.H, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)


class train_loader_multiscale(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, H, scale_num, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []
        self.H = H      
        self.scale_num = scale_num
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)               
        start = 0       
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end     

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')            
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))  
            visualFeatures.append(load_visual_multiscale(data, self.scale_num, self.visualPath,numFrames, self.H, visualAug = True))
            labels.append(load_label(data, numFrames))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)


class val_loader_multiscale(object):
    def __init__(self, trialFileName, audioPath, visualPath, H, scale_num, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.H = H  
        self.scale_num = scale_num
    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual_multiscale(data, self.scale_num, self.visualPath,numFrames, self.H, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)



class feat_extract_loader(object):
    def __init__(self, trialFileName, audioPath, visualPath, H, type=None):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.H = H  
        self.type = type
    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        clip_name = line[0].split('\t')[0]
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        if self.type == 'val':
            audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
            visualFeatures = [load_visual(data, self.visualPath,numFrames, self.H, visualAug = False)]
        if self.type == 'train':
            audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
            visualFeatures = [load_visual(data, self.visualPath,numFrames, self.H, visualAug = True)]
        labels = [load_label(data, numFrames)]         
        return clip_name, \
               torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)


class feat_extract_loader_multiscale(object):
    def __init__(self, scale_num, trialFileName, audioPath, visualPath, H, type=None):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.H = H  
        self.type = type
        self.scale_num = scale_num
    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        clip_name = line[0].split('\t')[0]
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        if self.type == 'val':
            audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
            visualFeatures = [load_visual_multiscale(data, self.scale_num, self.visualPath,numFrames, self.H, visualAug = False)]
        if self.type == 'train':
            audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
            visualFeatures = [load_visual_multiscale(data, self.scale_num, self.visualPath,numFrames, self.H, visualAug = True)]
        labels = [load_label(data, numFrames)]         
        return clip_name, \
               torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)



class train_loader_face_audio(object):
    def __init__(self, trialFileName, audioPath, visualPath, batchSize, H, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = []
        self.H = H      
        mixLst = open(trialFileName).read().splitlines()
        # sort the training set by the length of the videos, shuffle them to make more videos in the same batch belong to different movies
        sortedMixLst = sorted(mixLst, key=lambda data: (int(data.split('\t')[1]), int(data.split('\t')[-1])), reverse=True)               
        start = 0       
        while True:
            length = int(sortedMixLst[start].split('\t')[1])
            end = min(len(sortedMixLst), start + max(int(batchSize / length), 1))
            self.miniBatch.append(sortedMixLst[start:end])
            if end == len(sortedMixLst):
                break
            start = end     

    def __getitem__(self, index):
        batchList    = self.miniBatch[index]
        numFrames   = int(batchList[-1].split('\t')[1])
        audioFeatures, visualFeatures, labels = [], [], []
        audioSet = generate_audio_set(self.audioPath, batchList) # load the audios in this batch to do augmentation
        for line in batchList:
            data = line.split('\t')            
            audioFeatures.append(load_audio(data, self.audioPath, numFrames, audioAug = True, audioSet = audioSet))  
            visualFeatures.append(load_visual_face(data, self.visualPath,numFrames, self.H, visualAug = True))
            labels.append(load_label(data, numFrames))
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))        

    def __len__(self):
        return len(self.miniBatch)


class val_loader_face_audio(object):
    def __init__(self, trialFileName, audioPath, visualPath, H, **kwargs):
        self.audioPath  = audioPath
        self.visualPath = visualPath
        self.miniBatch = open(trialFileName).read().splitlines()
        self.H = H  
    def __getitem__(self, index):
        line       = [self.miniBatch[index]]
        numFrames  = int(line[0].split('\t')[1])
        audioSet   = generate_audio_set(self.audioPath, line)        
        data = line[0].split('\t')
        audioFeatures = [load_audio(data, self.audioPath, numFrames, audioAug = False, audioSet = audioSet)]
        visualFeatures = [load_visual_face(data, self.visualPath,numFrames, self.H, visualAug = False)]
        labels = [load_label(data, numFrames)]         
        return torch.FloatTensor(numpy.array(audioFeatures)), \
               torch.FloatTensor(numpy.array(visualFeatures)), \
               torch.LongTensor(numpy.array(labels))

    def __len__(self):
        return len(self.miniBatch)


class GraphDataset(Dataset):
    def __init__(self, args, mode = 'train', specify_graph=0):
        super(GraphDataset, self).__init__()
        if specify_graph == 0:
            self.data_path = os.path.join(args.graph_path, 'graph_'+str(args.numv)+'_'+str(args.time_edge),mode,'processed')
        else:
            self.data_path = os.path.join(args.graph_path, mode,'processed')
            
        self.all_files = os.listdir(self.data_path)

    @property
    def raw_file_names(self):
        return []

    def _download(self):
        return

    def _process(self):
        return 

    def process(self):
        return

    def len(self):
        return len(self.all_files)

    def get(self, idx):
        data_stack = torch.load(os.path.join(self.data_path, self.all_files[idx]))
        return data_stack
