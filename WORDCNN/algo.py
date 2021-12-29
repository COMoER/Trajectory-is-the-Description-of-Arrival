import torch
import torch.nn as nn
import pickle as pkl
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
HIDDEN_SIZE = 256
from tqdm import tqdm
from sklearn.metrics import accuracy_score

class mlp(nn.Module):
    def __init__(self,cls):
        super(mlp, self).__init__()
        self.fc1 = nn.Linear(32,HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE,cls)
    def forward(self,x):
        l2 = F.relu(self.fc1(x))
        return self.fc2(l2)

class dataset(Dataset):
    def __init__(self,X,y):

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return self.y.shape[0]
    def __getitem__(self, idx):
        return self.X[idx],self.y[idx]
if __name__ == '__main__':
    model = mlp(56961)
    optim = torch.optim.Adam(model.parameters(),0.001)
    critic = torch.nn.CrossEntropyLoss()
    with open("../train_data.pth", 'rb') as f:
        X, y = pkl.load(f)
    y += 1
    size = len(X)
    dataloader = DataLoader(dataset(X[:round(size*0.95)],y[:round(size*0.95)]),32,True,drop_last=True)
    X_test = torch.FloatTensor(X[round(size*0.95):])
    y_test = torch.LongTensor(y[round(size*0.95):])
    for epoch in range(5):
        acc = 0
        n = 0
        tbar = tqdm(dataloader)
        for X,y in tbar:
            pred = model(X)
            loss = critic(pred,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tbar.set_postfix({'loss':"%.3f"%loss.cpu().detach().item()})
        logits = model(X_test).detach()
        pred = torch.argmax(logits,dim=1)
        print(f"epoch {epoch} loss is {critic(logits,y_test).detach().cpu().item():.3f} acc is {accuracy_score(pred,y_test):.3f}")
    torch.save(model.state_dict,"model.pt")

