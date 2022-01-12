from pathlib import Path
import sys
import os
import argparse
sys.path.append(str(Path(__file__).resolve().parent.parent))
install_path = str(Path(__file__).resolve().parent.parent)
dataset_path = str(Path(__file__).resolve().parent.parent.parent)

import torch
import torch.nn as nn

import pandas as pd
import pickle as pkl
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model.algo_first_point import mlpMetaEmbedding,device,TrajectoryDataset
from utils.common import loadGeoHashEncoder,LonLatVocal
from utils.coordination import LatLonTransform,loadTransform
from utils.preprocess import pipeline
from utils.meta import getEmbedInfo

modelGeo:LonLatVocal = loadGeoHashEncoder(os.path.join(install_path,"pretrained","geohashmodel.pt"))
N = len(modelGeo) + 1
del modelGeo

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix",action="store_true",help="using 5 prefix input")
    parser.add_argument("--meta",action="store_true",help="add metadata to train")
    args = parser.parse_args()
    metainfo = getEmbedInfo() if args.meta else None
    model_em = mlpMetaEmbedding(N, meta=args.meta, first_point=args.prefix, embed_table=metainfo).to(device).eval()
    model_em.load_state_dict(torch.load(os.path.join(install_path,"pretrained","model_best.pt"), device))
    df = pd.read_csv(os.path.join(dataset_path,"dataset","test.csv"), delimiter=',')
    X,first_points_test,meta_test = pipeline(df,test=True)
    pred = []
    for s,first_points,meta in zip(X,first_points_test,meta_test):
        x = torch.LongTensor(s + 1).view(1, -1).to(device) # plus 1 as the shift of padding
        first_points = torch.FloatTensor(first_points)
        meta = torch.LongTensor(meta).view(1,-1).to(device)
        a,b = first_points.shape
        first_points = first_points.view(1,a,b).to(device)
        y = model_em(x,first_points,meta).detach().cpu().numpy().reshape(-1)
        pred.append(y)

    pred = np.stack(pred, 0)
    trans = loadTransform(os.path.join(install_path,"pretrained","trans.pt"))
    pred = trans.detransform(pred)
    lat = pred[:, 0]
    lon = pred[:, 1]

    tra = df["TRIP_ID"]
    if not os.path.exists(os.path.join(install_path,"results")):
        os.mkdir(os.path.join(install_path,"results"))
    sub_df = pd.DataFrame(data=list(zip(tra, lat, lon)), columns=["TRIP_ID", "LATITUDE", "LONGITUDE"])
    sub_df.to_csv(os.path.join(install_path,"results","mysubmit.csv"), index=False)