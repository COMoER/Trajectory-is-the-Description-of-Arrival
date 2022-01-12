import torch
import torch.nn as nn
import pickle as pkl
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

HIDDEN_SIZE = 256
from tqdm import tqdm

import random
from utils.common import setup_seed

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


class RandomSelect(nn.Module):
    def __init__(self, length, T, head, test):
        super(RandomSelect, self).__init__()
        self.length = length
        self.T = T
        self.head = head
        self.test = test
        if test:
            # test dataset not random select length while different path
            self.l = random.randint(round(length * 0.1),round(length*0.5))
            self.start = 0

    def forward(self, paths):
        length = self.length
        if self.test:
            mask = torch.zeros((self.T,), dtype=torch.long)
            mask[self.start:self.start + self.l] = 1
            return paths * mask
        if self.head:
            l = random.randint(round(length * 0.3), round(length * 0.7))
            start = 0
        else:
            l = random.randint(round(length * 0.5), round(length * 0.9))
            start = random.randint(0, length - l + 1)
        mask = torch.zeros((self.T,), dtype=torch.long)
        mask[start:start + l] = 1
        return paths * mask


class TrajectoryDataset(Dataset):
    def __init__(self, sample, y, train_first_points, vocal_max,test=False,head=False,random=False):
        """
        Args:
            sample: List[path]
            arrival: List[node]
            train_first_points: List[position]
            vocal_max: len of geohash
            K: neg length
        """
        setup_seed(2021)
        super(TrajectoryDataset, self).__init__()
        self.vocal_max = vocal_max
        self.T = max([len(v) for v in sample])
        self.random = random
        self.sample = []
        self.randomSelect = []
        for sen in tqdm(sample):
            senpad = np.zeros(self.T, np.long)
            if self.random:
                self.randomSelect.append(RandomSelect(len(sen), self.T, head, test))
            senpad[:len(sen)] = np.array(sen) + 1  # shift one position to contain padding 0
            self.sample.append(senpad)
        self.arrival = y
        self.paths = torch.LongTensor(self.sample)  # (N,T)
        self.nodes = torch.FloatTensor(self.arrival).view(-1, 2)  # (N,2)
        self.n_point = torch.FloatTensor(train_first_points) # (N,5,2)
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        if self.random:
            return self.randomSelect[idx](self.paths[idx]), self.arrival[idx], self.n_point[idx]
        else:
            return self.paths[idx], self.arrival[idx], self.n_point[idx]


class mlp(nn.Module):
    def __init__(self, vocal_max):
        super(mlp, self).__init__()
        self.embed = nn.Embedding(vocal_max + 1, 32, padding_idx=0)  # padding
        self.conv = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x,first_points): #first point B,5,2
        embedding = self.embed(x)  # B,T,32
        embedding = embedding.permute(0, 2, 1)  # B,C,T
        l1 = F.relu(self.conv(embedding))
        l1_max = torch.max(l1, dim=-1)[0]  # B,32 over-time-maxpooling
        l2 = self.dropout(F.relu(self.fc1(l1_max)))
        return self.fc2(l2)



