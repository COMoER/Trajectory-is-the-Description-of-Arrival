from pathlib import Path
import sys
sys.path.append(Path(__file__).resolve().parent.parent)

import pandas as pd
import numpy as np
import pickle as pkl
from utils.common import str2trajectory, loadGeoHashEncoder, LonLatVocal
from utils.coordination import loadTransform, LatLonTransform


def pipeline(df, n=100000,test=False,verbose=True):

    modelGeo: LonLatVocal = loadGeoHashEncoder("pretrained/geohashmodel.pt")

    if test:
        se, mask = str2trajectory(df, return_mask=True)
        seq, mask_seq = modelGeo.transform(se, arrival=False, return_mask=True, test=True)
        X = seq
        train_first_points = []
        n_first_point = 5 #select first 5 point
        trans:LatLonTransform = loadTransform("pretrained/trans.pt")
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



        return X,train_first_points

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

    arrival = np.array([seq[-1] for seq in whole_seq])
    starter = np.array([seq[0] for seq in whole_seq])
    train_seq = np.array([seq[:-1] for seq in whole_seq])
    train_latlon_seq = []
    # get the arrival in lat-lon coordination
    for m, seq in zip(mask, series):
        if m:
            train_latlon_seq.append((seq[-1], seq[-2]))
    train_first_points = []
    n_first_point = 5 #select first 5 point
    trans:LatLonTransform = loadTransform("pretrained/trans.pt")
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
    sample_df["arr"] = train_latlon_seq
    sample_df["arrival"] = arrival
    sample_df["starter"] = starter
    sample_df["seq"] = train_seq

    X = sample_df['seq'].to_list()
    y = np.array(sample_df['arr'].to_list())

    # transform from lat-lon coordination to x-y coordination
    y = trans.transform(y)
    if verbose:
        print()
    return X, y ,train_first_points
