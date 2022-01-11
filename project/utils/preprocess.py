from pathlib import Path
import sys
import os
sys.path.append(Path(__file__).resolve().parent.parent)
install_path = str(Path(__file__).resolve().parent.parent)

import pandas as pd
import numpy as np
import pickle as pkl
from utils.common import str2trajectory, loadGeoHashEncoder, LonLatVocal,setup_seed
from utils.coordination import loadTransform, LatLonTransform
from utils.meta import mapMeta


def pipeline(df, n=100000,test=False,verbose=True):

    modelGeo: LonLatVocal = loadGeoHashEncoder(os.path.join(install_path,"pretrained","geohashmodel.pt"))

    if test:
        df,meta_index = mapMeta(df)
        meta_data = []
        for meta in meta_index:
            meta_data.append(df[meta].tolist())
        se, mask = str2trajectory(df, return_mask=True)
        seq, mask_seq = modelGeo.transform(se, arrival=False, return_mask=True, test=True)
        X = seq
        train_first_points = []
        n_first_point = 5 # select first 5 point
        trans:LatLonTransform = loadTransform(os.path.join(install_path,"pretrained","trans.pt"))
        for m, seq in zip(mask, se):
            if m:
                lon = seq[::2]
                lat = seq[1::2]
                position = np.vstack((lat,lon))
                position = position.T
                if (position.shape[0]<n_first_point):
                    padding=np.tile(position[-1],(n_first_point-position.shape[0],1))
                    position = np.vstack((position,padding))
                position = position[:n_first_point]
                position_xy=trans.transform(position)
                train_first_points.append(position_xy)

            return X,train_first_points,np.array(meta_data).T
    setup_seed(2021)
    # from dataset sample n samples
    sample_df = df.sample(n)
    # preprocess of the dataset str trajectory
    series, mask = str2trajectory(sample_df, return_mask=True)

    if verbose:
        print('.',end='')

    sample_df = sample_df[mask]
    # transform from lonlat to geohash
    whole_seq, mask = modelGeo.transform(series, return_mask=True)

    if verbose:
        print('.',end='')

    train_seq = np.array([seq[:-1] for seq in whole_seq])
    train_latlon_seq = []
    # get the arrival in lat-lon coordination
    for m, seq in zip(mask, series):
        if m:
            train_latlon_seq.append((seq[-1], seq[-2]))
    train_first_points = []
    n_first_point = 5 #select first 5 point
    trans:LatLonTransform = loadTransform(os.path.join(install_path,"pretrained","trans.pt"))
    for m, seq in zip(mask, series):
        if m:
            seq = seq[:-2]
            lon = seq[::2]
            lat = seq[1::2]
            position = np.vstack((lat,lon))
            position = position.T
            if (position.shape[0]<n_first_point):
                padding=np.tile(position[-1],(n_first_point-position.shape[0],1))
                position = np.vstack((position,padding))
            position = position[:n_first_point]
            position_xy=trans.transform(position)
            train_first_points.append(position_xy)

    if verbose:
        print('.',end='')

    sample_df = sample_df[mask]
    sample_df,meta_index = mapMeta(sample_df)
    meta_data = []
    for meta in meta_index:
        meta_data.append(sample_df[meta].tolist())

    X = train_seq
    y = np.array(train_latlon_seq)

    # transform from lat-lon coordination to x-y coordination
    y = trans.transform(y)
    if verbose:
        print()
    return X, y ,train_first_points,np.array(meta_data).T
