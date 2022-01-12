from pathlib import Path
import sys
import os
sys.path.append(Path(__file__).resolve().parent.parent)
install_path = str(Path(__file__).resolve().parent.parent)
import numpy as np
from utils.coordination import LatLonTransform,loadTransform

def haversine(x,y):
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    x = np.deg2rad(x)
    y = np.deg2rad(y)

    # haversine formula
    dlon = y[:,1] - x[:,1]
    dlat = y[:,0] - x[:,0]
    a = np.sin(dlat/2)**2 + np.cos(x[:,0]) * np.cos(y[:,0]) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def test_score(pred,gt):
    """
    Args:
        pred: [N,2] numpy (x,y)
        y: [N,2] numpy (x,y)
    """
    trans = loadTransform(os.path.join(install_path,"pretrained","trans.pt"))
    pred = trans.detransform(pred) # N,2 lat,lon
    gt = trans.detransform(gt) # N,2 lat,lon
    return np.mean(haversine(pred,gt))