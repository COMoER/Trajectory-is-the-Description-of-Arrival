import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import random

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)

train_vali_radio = 0.9

class skipgram(nn.Module):
    def __init__(self, N, D):
        super(skipgram, self).__init__()
        self.D = D
        self.embed = nn.Embedding(N+1, D) # including padding

    def forward(self, v, paths, negs,mask):
        """
        Args:
            v: Tensor (N,) Long
            paths: Tensor (N,L) Long
            negs:  Tensor (N,L,K) Long
            mask:  Tensor (N,L) Long
        Returns:

        """
        mask = mask.detach()
        N,L, K = negs.shape
        x = self.embed(paths) # N,L,D
        n = self.embed(negs).view(N*L,K,-1) # N*L,K,D
        v = self.embed(v).unsqueeze(2) #(N,D,1)
        path_logprob = torch.log(torch.sigmoid(torch.bmm(x, v))).squeeze(2) # N,L
        v = v.unsqueeze(1).expand(N,L,-1,1).reshape(N*L,-1,1) # N*L,D,1
        neg_logprob = torch.sum(torch.log(torch.sigmoid(-torch.bmm(n, v))).squeeze(2), dim=1).reshape(N,L) #N,L
        ww = - path_logprob - neg_logprob
        return torch.mean(torch.sum(ww*mask,1))


class edge_dataset(Dataset):
    def __init__(self,sample,arrival,vocal_max,K):
        """

        Args:
            sample: List[path]
            arrival: List[node]
            vocal_max: len of geohash
        """
        super(edge_dataset, self).__init__()
        self.vocal_max = vocal_max+1 # including padding 0
        self.T = max([len(v) for v in sample])
        self.sample = []
        self.length = []
        for sen in sample:
            senpad = np.zeros(self.T,np.long)
            mask = np.zeros(self.T,int)
            senpad[:len(sen)] = np.array(sen)+1
            mask[:len(sen)] = 1
            self.sample.append(senpad)
            self.length.append(mask)
        self.arrival = arrival
        self.K = K
        self.paths = torch.LongTensor(self.sample).to(device)  # (N,T)
        self.nodes = torch.LongTensor(self.arrival).to(device).view(-1)  # (N,)
        self.length = torch.FloatTensor(self.length).to(device).view(-1,self.T)  # (N,T)

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, idx):
        neg = torch.randint(1,self.vocal_max+1, (self.T,self.K),dtype=torch.long).to(device)
        return neg,self.paths[idx],self.nodes[idx],self.length[idx]


class deepwork:
    def __init__(self, d: int, n:int,gamma: int,  K: int, lr: float):
        """
        to calc in_deg out_deg to every vertices
        Args:
            d: word vec length
            gamma:walks per vertex
            t:random walk times
            K: negative sampling size
            lr:learning rate
        """
        self.gamma = gamma
        self.d = d
        self.K = K
        self.lr = lr
        self.net = skipgram(n+1, self.d).to(device)
        self.optim = torch.optim.SGD(self.net.parameters(), self.lr)
        self.LS = torch.optim.lr_scheduler.StepLR(self.optim,5)

    def load_data(self, sample,arrival,vocal_max):
        """
        Args:
            edges: np.ndarray (N,(src,dst,weight))
            log_dir: the dir to save the validation_set
        """
        self._dataset = edge_dataset(sample,arrival,vocal_max,self.K)

    def train(self, log_dir):
        dataloader = DataLoader(self._dataset,batch_size=16,shuffle=True,drop_last=True)
        # divide validation and train
        for i in range(self.gamma):
            tbar = tqdm(dataloader)
            loss_sum = []
            for negs, paths, nodes,length in tbar:

                loss = self.net(nodes, paths, negs,length)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                loss = loss.item()
                loss_sum.append(loss)
                tbar.set_postfix({'loss':"%.3f"%loss_sum[-1]})
            # self.LS.step() # learning rate delay
            print("[INFO] epoch %d loss %.3f lr %.3f" % (i+1, float(np.mean(np.array(loss_sum))),self.LS.get_last_lr()[0]))
            if (i + 1) % 5 == 0:
                torch.save(self.net.state_dict(), log_dir / Path("result_%d.pt" % (i + 1)))

    def eval(self, filename):
        ckpt = torch.load(filename, device)
        self.net.load_state_dict(ckpt)
