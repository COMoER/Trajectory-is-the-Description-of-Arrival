from pathlib import Path
import sys
sys.path.append(Path(__file__).resolve().parent.parent)

import torch
import torch.nn as nn
import pickle as pkl
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model.algo import mlp,device,TrajectoryDataset
from utils.common import loadGeoHashEncoder,LonLatVocal
from utils.coordination import LatLonTransform
from utils.preprocess import pipeline

modelGeo:LonLatVocal = loadGeoHashEncoder("../pretrained/geohashmodel.pt")
N = len(modelGeo) + 1
del modelGeo

if __name__ == '__main__':
    model = mlp(N).to(device)
    optim = torch.optim.Adam(model.parameters(),0.01)
    critic = torch.nn.MSELoss()
    print("[INFO] load dataset")
    df_train = pd.read_csv("../../dataset/train.csv",delimiter=',')
    print("[INFO] doing preprocessing")
    X,y = pipeline(df_train,100000,verbose=True)
    size = len(X)
    print("[INFO] doing preprocessing finished")
    train_dataloader = DataLoader(TrajectoryDataset(X[:round(size*0.95)],y[:round(size*0.95)],56960),200,True,drop_last=True)
    test_data = TrajectoryDataset(X[round(size*0.95):],y[round(size*0.95):],56960)
    test_dataloader = DataLoader(test_data,len(test_data))
    print("[INFO] data load finish")
    X_test,y_test = None,None
    for x,y in test_dataloader:
        X_test = x.to(device)
        y_test = y.to(device)
    error_best = 100
    for epoch in range(100):
        acc = 0
        n = 0
        tbar = tqdm(train_dataloader)
        model.train()
        for X,y in tbar:
            pred = model(X.to(device))
            loss = critic(pred,y.to(device).float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            tbar.set_postfix({'loss':"%.3f"%loss.cpu().detach().item()})
        model.eval()
        pred = model(X_test).detach()
        err = critic(pred,y_test).cpu().detach().item()
        print(f"epoch {epoch} mse is {err:.3f}")
        if err<error_best:
            error_best = err
            torch.save(model.state_dict(),"model_best.pt")