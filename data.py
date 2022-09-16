import os
import numpy as np
from scipy import signal
import torch
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor

# need to be modified!

def transform():
    return Compose([
        ToTensor(),
    ])


def getSTMap(info):
    data_info=info.split('.')
    index=int(data_info[1])
    path_ori='/home/som/8T/kz/paper1/MMVS/'

    path_stmap_v=os.path.join(path_ori,data_info[0],'STMap.npy')
    stmap_v =np.load(path_stmap_v)[:,(3+2*index)*25:(13+2*index)*25]
    path_stmap_i = os.path.join(path_ori, data_info[0], 'STMap_infrared.npy')
    stmap_i = np.load(path_stmap_i)[:, (3 + 2 * index) * 25:(13 + 2 * index) * 25]

    path_ppg=os.path.join(path_ori,data_info[0],'PPG_Signal.npy')
    ppg = np.load(path_ppg)[(3+2*index)*60:(13+2*index)*60]
    ppg = (ppg - np.mean(ppg)) / np.std(ppg)
    ppg = signal.resample(ppg,250)

    return stmap_v,stmap_i,ppg

class DatasetFromFolder(data.Dataset):
    def __init__(self,datapath,transform=transform()):
        super().__init__()
        self.data = np.load(datapath)
        self.transform = transform

    def __getitem__(self, index):
        # my data sample [path, hr, sbp, dbp,rr,spo2]
        # ['20201021/21_1.2' '63.0' '133' '81' '23.81' '98.0']
        info = self.data[index]
        stmap_v,stmap_i,ppg = getSTMap(info[0])
        stmap_v = self.transform(stmap_v)
        stmap_i = self.transform(stmap_i)
        ppg = torch.tensor(ppg, dtype=torch.float32)
        hr = torch.tensor(float(info[1]), dtype=torch.float32)
        rr = torch.tensor(float(info[5]), dtype=torch.float32)
        return stmap_v,stmap_i,hr,ppg,rr

    def __len__(self):
        return len(self.data)
