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

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


setup_seed(2021)


class RandomSelect(nn.Module):
    def __init__(self, length, T, head,test):
        super(RandomSelect, self).__init__()
        self.length = length
        self.T = T
        self.head = head
        self.test = test
        if test:
            # test dataset not random select length while different path
            self.l = random.randint(round(length * 0.2),round(length*0.9))
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
    def __init__(self, sample, y,vocal_max,meta_data=None,train_first_points=None,
                 test=False,head=False,random=False,prefix=False,meta=False):
        """

        Args:
            sample: List[path]
            arrival: List[node]
            train_first_points: List[position]
            meta_data =
            vocal_max: len of geohash
            K: neg length
        """
        super(TrajectoryDataset, self).__init__()
        self.vocal_max = vocal_max
        self.T = max([len(v) for v in sample])
        self.random = random
        self.prefix = prefix
        self.meta = meta
        self.sample = []
        self.randomSelect = []
        for sen in tqdm(sample):
            senpad = np.zeros(self.T, np.long)
            if self.random:
                self.randomSelect.append(RandomSelect(len(sen), self.T,head,test))
            senpad[:len(sen)] = np.array(sen) + 1  # shift one position to contain padding 0
            self.sample.append(senpad)
        self.arrival = y
        self.paths = torch.LongTensor(self.sample)  # (N,T)
        self.nodes = torch.FloatTensor(self.arrival).view(-1, 2)  # (N,2)
        self.meta_data = torch.LongTensor(meta_data) # N,M
        self.n_point = torch.FloatTensor(train_first_points) # (N,5,2)
    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        if self.random:
            return self.randomSelect[idx](self.paths[idx]), self.arrival[idx], self.n_point[idx],self.meta_data[idx] 
        else:
            return self.paths[idx], self.arrival[idx], self.n_point[idx],self.meta_data[idx]



class outputLayer(nn.Module):
    def __init__(self, input_size):
        super(outputLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        l2 = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(l2)


class metaEmbedding(nn.Module):
    def __init__(self, embed_table: dict, norm=True):
        super(metaEmbedding, self).__init__()
        self.embed_table = nn.ModuleList()
        self.dim = 0
        self.norm = norm
        for _, info in embed_table.items():
            num, size, pad = info
            self.dim += size
            if pad:
                self.embed_table.append(nn.Embedding(num + 1, size, padding_idx=0))
            else:
                self.embed_table.append(nn.Embedding(num, size))
    @property
    def shape(self):
        return self.dim
    def forward(self, metas):
        embeddings = []
        for i, layer in enumerate(self.embed_table):
            v = layer(metas[:,i])
            if self.norm:
                n = torch.norm(v, dim=-1, keepdim=True)
                n[torch.isclose(n,torch.zero_(n))]=1
                v = v/n
            embeddings.append(v)
        return torch.cat(embeddings, -1)
class mlpMetaEmbedding(nn.Module):
    def __init__(self, vocal_max, meta=False, first_point=False,embed_table=None):
        super(mlpMetaEmbedding, self).__init__()
        self.embed = nn.Embedding(vocal_max + 1, 32, padding_idx=0)  # padding
        self.conv = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        N = 32
        if first_point:
            self.conv_first_point = nn.Conv1d(2, 16, kernel_size=3, padding=1)
            N += 16
        self.meta = meta
        self.first_point=first_point
        if meta:
            self.metaEmbed = metaEmbedding(embed_table)
            N += self.metaEmbed.shape
        self.out = outputLayer(N)
    def forward(self, x, first_points=None,metas=None):

        embedding = self.embed(x)  # B,T,32
        embedding = embedding.permute(0, 2, 1)  # B,C,T
        if self.first_point:
            first_points = first_points.permute(0, 2, 1)
        l1 = F.relu(self.conv(embedding))
        l1_max = torch.max(l1, dim=-1)[0]  # B,32 over-time-maxpooling
        features = l1_max
        if self.meta or self.first_point:
            l1_max_norm = l1_max/torch.norm(l1_max,dim=-1,keepdim=True)
            features = l1_max_norm
        if self.first_point:
            l1_first_point = F.relu(self.conv_first_point(first_points))
            l1_max_first_point = torch.max(l1_first_point, dim=-1)[0]
            l1_max_first_point_norm = l1_max_first_point / torch.norm(l1_max_first_point, dim=-1, keepdim=True)
            features = torch.cat((features, l1_max_first_point_norm), dim=1)  # B,32+16
        if self.meta:
            meta_embed = self.metaEmbed(metas)
            features = torch.cat([features,meta_embed],dim=1)
        return self.out(features)
