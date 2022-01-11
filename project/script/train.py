from pathlib import Path
import sys
import os
import logging
import argparse
from datetime import datetime
import yaml
sys.path.append(str(Path(__file__).resolve().parent.parent))
install_path = str(Path(__file__).resolve().parent.parent)
dataset_path = str(Path(__file__).resolve().parent.parent.parent)

import torch
import torch.nn as nn
import pickle as pkl
import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


from model.algo_first_point import mlpMetaEmbedding,device,TrajectoryDataset
from utils.common import loadGeoHashEncoder,LonLatVocal,setup_seed
from utils.coordination import LatLonTransform
from utils.preprocess import pipeline
from utils.meta import getEmbedInfo

modelGeo:LonLatVocal = loadGeoHashEncoder(os.path.join(install_path,"pretrained","geohashmodel.pt"))
N = len(modelGeo) + 1
del modelGeo

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_length",action="store_true",default=True,help="using random pick training")
    parser.add_argument("--head",action="store_true",help="using random prefix")
    parser.add_argument("--prefix",action="store_true",help="using 5 prefix input")
    parser.add_argument("--meta",action="store_true",help="add metadata to train")
    parser.add_argument("--out",action="store_true",help="add train data that has outlier geohash location")
    parser.add_argument("--lr",type=float,default=0.01,help="learning rate")
    parser.add_argument("--size",type=int,default=100000,help="# of randomly picked from trainset")
    parser.add_argument("--epoch",type=int,default=50,help="# of max epoch")
    args = parser.parse_args()

    dir = os.path.join(install_path,"log",datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(os.path.join(dir,"args.yaml"),'w') as f:
        ym = yaml.dump(vars(args),f)
    log_fname = os.path.join(dir, 'log_train.txt')
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
    logger = logging.getLogger("Trainer")

    metainfo = getEmbedInfo() if args.meta else None
    model = mlpMetaEmbedding(N,meta=args.meta, first_point=args.prefix,embed_table=metainfo).to(device)
    optim = torch.optim.Adam(model.parameters(),args.lr)
    critic = torch.nn.MSELoss()
    print("[INFO] load dataset")
    df_train = pd.read_csv(os.path.join(dataset_path,"dataset","train.csv"),delimiter=',')
    print("[INFO] doing preprocessing")
    X,y,first_points,meta_data = pipeline(df_train,args.size,verbose=True)

    #with open("../../train_data.pth",'rb') as f:
        #X,y = pkl.load(f)

    size = len(X)
    print("[INFO] doing preprocessing finished")
    # sample, y,vocal_max,meta_data=None,train_first_points=None,
    # test=False,head=False,random=False,prefix=False,meta=False
    train_dataloader = DataLoader(TrajectoryDataset(X[:round(size*0.95)],y[:round(size*0.95)],
                                                    56960,
                                                    meta_data=meta_data[:round(size*0.95)] if args.meta else None,
                                                    train_first_points = first_points[:round(size*0.95)] if args.prefix else None,
                                                    test=False,
                                                    head=args.head,
                                                    random=args.random_length,prefix=args.prefix,
                                                    meta=args.meta),300,True,drop_last=True)
    test_data = TrajectoryDataset(X[round(size*0.95):],y[round(size*0.95):],
                                                    56960,
                                                    meta_data=meta_data[round(size*0.95):] if args.meta else None,
                                                    train_first_points = first_points[round(size*0.95):] if args.prefix else None,
                                                    test=True,
                                                    head=args.head,
                                                    random=args.random_length,prefix=args.prefix,
                                                    meta=args.meta)
    test_dataloader = DataLoader(test_data,len(test_data))
    print("[INFO] data load finish")
    # X_test,y_test,first_points_test,meta_data_test = None,None,None,None
    for x,y,first_points,meta_data in test_dataloader:
        X_test = x.to(device)
        y_test = y.to(device)
        if first_points is not None:
            first_points_test = first_points.to(device)
        if meta_data is not None:
            meta_data_test = meta_data.to(device)
    error_best = 100
    setup_seed(2021)
    for epoch in range(args.epoch):
        logger.info('**** EPOCH %03d ****' % (epoch))
        acc = 0
        n = 0
        tbar = tqdm(train_dataloader)
        model.train()
        loss_sum = 0
        for X,y,first_points,meta_data in tbar:
            if first_points is not None:
                first_points = first_points.to(device)
            if meta_data is not None:
                meta_data = meta_data.to(device)
            pred = model(X.to(device),first_points,meta_data)
            loss = critic(pred,y.to(device).float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += loss.cpu().detach().item()
            tbar.set_postfix({'loss':"%.3f"%loss.cpu().detach().item()})

        model.eval()
        pred = model(X_test,first_points_test,meta_data_test).detach()
        err = critic(pred,y_test).cpu().detach().item()
        print(f"epoch {epoch} mse is {err:.3f}")
        logger.info('train_loss: %.6f',loss_sum/len(tbar))
        logger.info('eval_loss: %.6f',err)
        if err<error_best:
            error_best = err
            print(f"[BEST] epoch {epoch} mse is {err:.3f}")
            torch.save(model.state_dict(),os.path.join(dir,"model_best.pt"))
    torch.save(model.state_dict(),os.path.join(dir,"model_last.pt"))