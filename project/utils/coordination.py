from pathlib import Path
import sys
sys.path.append(Path(__file__).resolve().parent.parent)

import numpy as np
from utils.common import str2trajectory
import pickle as pkl

class LatLonTransform:
    def __init__(self,se_latlon):
        e = 0.0818191908426
        a = 6378137.0/1000
        mu = se_latlon[:,0]
        lam = se_latlon[:,1]
        mu = np.deg2rad(mu)
        lam = np.deg2rad(lam)
        self.mu_0 = np.min(mu)
        self.lam_0 = np.min(lam)
        self.Rn = a*(1-e**2)/(1-e**2*np.sin(self.mu_0)**2)**(3/2)
        self.Re = a/(1-e**2*np.sin(self.mu_0)**2)**(1/2)
    def transform(self,se_latlon):
        """
        se_latlon [N,2]
        mu LATITUDE 纬度, lam LONGITUDE 经度
        """
        mu = se_latlon[:,0]
        lam = se_latlon[:,1]
        mu = np.deg2rad(mu)
        lam = np.deg2rad(lam)

        x = np.cos(self.mu_0)*self.Re*(lam-self.lam_0)
        y = self.Rn*(mu-self.mu_0)

        return np.stack([x,y],1)
    def detransform(self,se_xy):
        x = se_xy[:,0]
        y = se_xy[:,1]
        lat = y/self.Rn+self.mu_0
        lon = x/self.Re/np.cos(self.mu_0)+self.lam_0
        lat = np.rad2deg(lat)
        lon = np.rad2deg(lon)
        return np.stack([lat,lon],1)

def trainTransform(sample_df):
    series = str2trajectory(sample_df, -1)
    se = np.concatenate(series)
    se_mu = se[1::2]
    se_lam = se[::2]
    se_mulam = np.stack([se_mu, se_lam], 1)
    trans = LatLonTransform(se_mulam)
    with open('trans.pt', 'wb') as f:
        pkl.dump(trans, f)

def loadTransform(path="trans.pt"):
    with open(path, 'rb') as f:
        trans = pkl.load(f)
    return trans