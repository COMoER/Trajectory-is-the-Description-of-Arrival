
import pandas as pd
import numpy as np

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

#%% md

##### Geohash

#%%

import numpy as np
_base32 = np.array(list('0123456789bcdefghjkmnpqrstuvwxyz'))
class GeoHash:
    def __init__(self):
        pass

    @staticmethod
    def _encode(lat, lon, mid_lat, mid_lon, pre, p,hashn):
        if pre==-1:
            return
        hash_level_lat = 1 << (2 * pre)
        hash_level_lon = 1 << (2 * pre+1)
        mask_lat = lat > mid_lat
        mask_lon = lon > mid_lon
        hashn += hash_level_lat * mask_lat + hash_level_lon * mask_lon
        mid_lat = mid_lat + (mask_lat * 2 - 1) * (180/(1<<p))
        mid_lon = mid_lon + (mask_lon * 2 - 1) * (360/(1<<p))

        GeoHash._encode(lat, lon, mid_lat, mid_lon, pre-1, p+1,hashn)
    @staticmethod
    def _d2c(hashn):
        hashn = hashn.copy()
        str_hash = np.full(hashn.shape[0],'',dtype=object)
        while (hashn>0).any():
            hash1 = list(hashn%32)
            str_hash = _base32[hash1] + str_hash
            hashn = hashn//32
        return str_hash
    @staticmethod
    def encode(se: np.ndarray, pre=12,return_str = False):
        """
        Args:
            se:np.ndarray [lon,lat] mu LATITUDE 纬度, lam LONGITUDE 经度
        Return:
            geohash
        """
        se_lat = se[1::2] + 90
        se_lon = se[::2] + 180

        hashn = np.zeros(se.shape[0]//2,dtype=np.int64)
        mid_lat = np.full(se.shape[0]//2,90)
        mid_lon = np.full(se.shape[0]//2,180)
        GeoHash._encode(se_lat,se_lon,mid_lat,mid_lon,pre*5//2-1,2,hashn)
        if return_str:
            return hashn,GeoHash._d2c(hashn)
        else:
            return hashn
    @staticmethod
    def _decode(lon,lat,seq,p,pre):
        if pre==-1:
            return
        lat_mask = (1<<(2*pre) & seq > 0)*2-1
        lon_mask = (1<<(2*pre+1) & seq > 0)*2-1
        lon += lon_mask*(180/(1<<p))
        lat += lat_mask*(90/(1<<p))
        GeoHash._decode(lon,lat,seq,p+1,pre-1)

    @staticmethod
    def decode(seq,pre=12):
        """
        Args:
            seq: (N,) encode
        """
        lon = np.full(seq.shape[0],180.)
        lat = np.full(seq.shape[0],90.)
        GeoHash._decode(lon,lat,seq,1,pre*5//2-1)
        lon -= 180
        lat -= 90
        return np.stack([lon,lat],1).reshape(-1)

#%% md

#### Using GeoHash to cluster

#%%

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
    def transform(self,series,arrival=False,return_mask=False):
        series_trans = []
        mask = []
        for se in series:
            x = GeoHash.encode(se,GEOHASHLEVEL)
            x = x-self._wordMin
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