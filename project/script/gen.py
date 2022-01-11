from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

import pandas as pd
import pickle as pkl
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model.algo_first_point import mlp,device,TrajectoryDataset
from utils.common import loadGeoHashEncoder,LonLatVocal
from utils.coordination import LatLonTransform,loadTransform
from utils.preprocess import pipeline

modelGeo:LonLatVocal = loadGeoHashEncoder("pretrained/geohashmodel.pt")
N = len(modelGeo) + 1
del modelGeo

if __name__ == '__main__':
    model_em = mlp(N).to(device).eval()
    model_em.load_state_dict(torch.load("pretrained/model_last.pt", device))
    df = pd.read_csv("dataset/test.csv", delimiter=',')
    X,first_points_test = pipeline(df,test=True)
    pred = []
    for s,first_points in zip(X,first_points_test):
        x = torch.LongTensor(s + 1).view(1, -1).to(device) # plus 1 as the shift of padding
        first_points = torch.FloatTensor(first_points)
        a,b = first_points.shape
        first_points = first_points.view(1,a,b).to(device)
        y = model_em(x,first_points).detach().cpu().numpy().reshape(-1)
        pred.append(y)

    pred = np.stack(pred, 0)
    trans = loadTransform("pretrained/trans.pt")
    pred = trans.detransform(pred)
    lat = pred[:, 0]
    lon = pred[:, 1]

    tra = df["TRIP_ID"]

    sub_df = pd.DataFrame(data=list(zip(tra, lat, lon)), columns=["TRIP_ID", "LATITUDE", "LONGITUDE"])
    sub_df.to_csv("results/mysubmit.csv", index=False)