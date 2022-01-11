from pathlib import Path
import sys
import os
sys.path.append(Path(__file__).resolve().parent.parent)
install_path = str(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import pickle as pkl
from utils.geohash import GeoHash
import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
def str2trajectory(df,n=-1,return_mask=False):
    series = []
    mask = []
    if n>=0:
        df_new = df.sample(n)
        for flag,j in zip(df_new["MISSING_DATA"],df_new["POLYLINE"]):
            try:
                assert not flag
                series.append(np.array(j.replace('[','').replace(']','').split(','),float))
            except:
                pass
    else:
        for flag,j in zip(df["MISSING_DATA"],df["POLYLINE"]):
            try:
                assert flag==False
                series.append(np.array(j.replace('[','').replace(']','').split(','),float))
                mask.append(True)
            except:
                mask.append(False)
    if return_mask and n<0:
        return series,np.array(mask)
    else:
        return series


GEOHASHLEVEL = 7
class LonLatVocal:
    def __init__(self):
        self._wordMin = 0
        self._wordMax = 0
    def __len__(self):
        return self._wordMax
    def fit(self,df,n=100000,quantile=[0.01,0.99],verbose=False):
        series = str2trajectory(df,n)
        se = np.concatenate(series)
        lat = se[1::2]
        lon = se[::2]
        def getQmask(x):
            return np.logical_and(x<=np.quantile(x,quantile[1]),x>=np.quantile(x,quantile[0]))
        mask = np.logical_and(getQmask(lat),getQmask(lon))
        se_mask = np.stack([lon[mask],lat[mask]],1).reshape(-1)
        x = GeoHash.encode(se_mask,GEOHASHLEVEL)
        if verbose:
            print(f"Origin size {len(se):d}, [{quantile[0]},{quantile[1]}] {len(se_mask):d}",)
        self._wordMin = x.min()
        x_short = x-self._wordMin
        if verbose:
            print(f"reserved class size {max(x_short)}")
        self._wordMax = x_short.max()
    def transform(self,series,arrival=False,return_mask=False,test=False):
        series_trans = []
        mask = []
        for se in series:
            x = GeoHash.encode(se,GEOHASHLEVEL)
            x = x-self._wordMin
            if test:
                x[np.logical_or(x > self._wordMax,x < 0)] = -1 # padding
                series_trans.append(x[-1] if arrival else x)
                mask.append(True)
                continue
            if (x <= self._wordMax).all() and (x >= 0).all():
                if arrival or len(x) > 1:
                    series_trans.append(x[-1] if arrival else x)
                    mask.append(True)
                else:
                    mask.append(False)
            else:
                mask.append(False)
        if return_mask:
            return series_trans,np.array(mask)
        else:
            return series_trans
    def decode(self,trans_se,using_list=True):
        """
        Args:
            trans_se: (N,M) or (N,)
        """
        assert not using_list and trans_se.ndim==1
        if using_list:
            decode_se = []
            for se in trans_se:
                se += self._wordMin
                decode_se.append(GeoHash.decode(se,GEOHASHLEVEL))
        else:
            decode_se = GeoHash.decode(trans_se+self._wordMin,GEOHASHLEVEL)
        return decode_se

def trainGeoHash(df:pd.DataFrame,n=100000):
    """
    train GeoHash Model
    """
    model = LonLatVocal()
    model.fit(df[:n], n=-1, verbose=True)
    with open(os.path.join(install_path,"pretrained","geohashmodel.pt"), 'wb') as f:
        pkl.dump(model, f)

def loadGeoHashEncoder(path="geohashmodel.pt"):
    with open(path, 'rb') as f:
        model = pkl.load(f)
    return model