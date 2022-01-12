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

from model.algo_first_point import mlpMetaEmbedding, device, TrajectoryDataset
from utils.common import loadGeoHashEncoder, LonLatVocal, setup_seed
from utils.metric import test_score
from utils.coordination import LatLonTransform
from utils.preprocess import pipeline
from utils.meta import getEmbedInfo

modelGeo: LonLatVocal = loadGeoHashEncoder(os.path.join(install_path, "pretrained", "geohashmodel.pt"))
N = len(modelGeo) + 1
del modelGeo


def dataset_split(X, y, first_points, meta_data):
    size = len(X)
    dataset_setting = {'train': [0, 0.85], 'val': [0.85, 0.90], 'test': [0.90, 1]}
    dataset = {}
    for name, (start, end) in dataset_setting.items():
        dataset[name] = [X[round(start * size):round(end * size)], y[round(start * size):round(end * size)],
                         first_points[round(start * size):round(end * size)],
                         meta_data[round(start * size):round(end * size)]]
    return dataset


setup_seed(2021)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_length", action="store_true", default=True, help="using random pick training")
    parser.add_argument("--head", action="store_true", help="using random prefix")
    parser.add_argument("--prefix", action="store_true", help="using 5 prefix input")
    parser.add_argument("--meta", action="store_true", default=False, help="add metadata to train")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--size", type=int, default=150000, help="# of randomly picked from trainset")
    parser.add_argument("--epoch", type=int, default=50, help="# of max epoch")
    args = parser.parse_args()

    dir = os.path.join(install_path, "log", datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(os.path.join(dir, "args.yaml"), 'w') as f:
        ym = yaml.dump(vars(args), f)
    log_fname = os.path.join(dir, 'log_train.txt')
    LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    DATE_FORMAT = '%Y%m%d %H:%M:%S'
    logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
    logger = logging.getLogger("Trainer")

    metainfo = getEmbedInfo() if args.meta else None
    model = mlpMetaEmbedding(N, meta=args.meta, first_point=args.prefix, embed_table=metainfo).to(device)
    optim = torch.optim.Adam(model.parameters(), args.lr)

    critic = torch.nn.MSELoss()
    print("[INFO] load dataset")
    df_train = pd.read_csv(os.path.join(dataset_path, "dataset", "train.csv"), delimiter=',')
    print("[INFO] doing preprocessing")
    X, y, first_points, meta_data = pipeline(df_train, args.size, verbose=True)
    del df_train

    # with open("../../train_data.pth",'rb') as f:
    # X,y = pkl.load(f)

    print("[INFO] doing preprocessing finished")

    dataset = dataset_split(X, y, first_points, meta_data)

    # if not os.path.exists(os.path.join(install_path, "pretrained", "test.path")):
    #     with open(os.path.join(install_path, "pretrained", "test.path"), 'wb') as f:
    #         pkl.dump(dataset['test'], f)

    train_dataloader = DataLoader(TrajectoryDataset(dataset['train'][0],
                                                    dataset['train'][1],
                                                    56960,
                                                    train_first_points=dataset['train'][2],
                                                    meta_data=dataset['train'][3],
                                                    test=False,
                                                    head=args.head,
                                                    random=args.random_length, prefix=args.prefix,
                                                    meta=args.meta), 300, True, drop_last=True)
    val_data = TrajectoryDataset(dataset['val'][0],
                                 dataset['val'][1],
                                 56960,
                                 train_first_points=dataset['val'][2],
                                 meta_data=dataset['val'][3],
                                 test=True,
                                 head=args.head,
                                 random=args.random_length, prefix=args.prefix,
                                 meta=args.meta)
    test_data = TrajectoryDataset(dataset['test'][0],
                                  dataset['test'][1],
                                 56960,
                                 train_first_points=dataset['test'][2],
                                 meta_data=dataset['test'][3],
                                 test=True,
                                 head=args.head,
                                 random=args.random_length, prefix=args.prefix,
                                 meta=args.meta)
    val_dataloader = DataLoader(val_data, len(val_data))
    test_dataloader = DataLoader(test_data, len(test_data))
    print("[INFO] data load finish")
    for x, y, first_points, meta_data in test_dataloader:
        X_test = x
        y_test = y
        if first_points is not None:
            first_points_test = first_points
        if meta_data is not None:
            meta_data_test = meta_data
    for x, y, first_points, meta_data in val_dataloader:
        X_val = x
        y_val = y
        if first_points is not None:
            first_points_val = first_points
        if meta_data is not None:
            meta_data_val = meta_data
    error_best = 100

    for epoch in range(args.epoch):
        logger.info('**** EPOCH %03d ****' % (epoch))
        acc = 0
        n = 0
        tbar = tqdm(train_dataloader)
        model.train()
        loss_sum = 0
        for X, y, first_points, meta_data in tbar:
            first_points = first_points.to(device)
            meta_data = meta_data.to(device)
            pred = model(X.to(device), first_points, meta_data)
            loss = critic(pred, y.to(device).float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_sum += loss.cpu().detach().item()
            tbar.set_postfix({'loss': "%.3f" % loss.cpu().detach().item()})

        model.eval()
        pred = model(X_val.to(device), first_points_val.to(device), meta_data_val.to(device)).detach()
        err = critic(pred, y_val.to(device)).cpu().detach().item()
        err_h = test_score(pred.cpu().numpy().reshape(-1,2),y_val.detach().numpy().reshape(-1,2))
        print(f"epoch {epoch} mse is {err:.3f} err is {err_h:.3f}km")
        logger.info('train_loss: %.6f', loss_sum / len(tbar))
        logger.info('val_loss: %.6f', err)
        logger.info('val_err: %.6fkm', err_h)
        if err < error_best:
            error_best = err
            print(f"[BEST] epoch {epoch} mse is {err:.3f} err is {err_h:.3f}km")
            torch.save(model.state_dict(), os.path.join(dir, "model_best.pt"))

    torch.save(model.state_dict(), os.path.join(dir, "model_last.pt"))
    model.eval()
    pred = model(X_test.to(device), first_points_test.to(device), meta_data_test.to(device)).cpu().detach().numpy().reshape(-1, 2)
    err_h = test_score(pred, y_test.detach().numpy().reshape(-1, 2))
    logger.info('last model test_err: %.6fkm', err_h)
    model.load_state_dict(torch.load(os.path.join(dir, "model_best.pt"),device))
    model.eval()
    pred = model(X_test, first_points_test, meta_data_test).cpu().detach().numpy().reshape(-1, 2)
    err_h = test_score(pred, y_test.detach().numpy().reshape(-1, 2))
    logger.info('best model test_err: %.6fkm', err_h)