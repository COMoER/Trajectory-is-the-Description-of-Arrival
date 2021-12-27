import numpy as np
_base32 = np.array(list('0123456789bcdefghjkmnpqrstuvwxyz'))
class GeoHash:
    def __init__(self):
        pass

    @staticmethod
    def _encode(lat, lon, mid_lat, mid_lon, pre, p,hash):
        if pre==-1:
            return
        hash_level_lat = 1 << (2 * pre)
        hash_level_lon = 1 << (2 * pre+1)
        mask_lat = lat > mid_lat
        mask_lon = lon > mid_lon
        hash += hash_level_lat * mask_lat + hash_level_lon * mask_lon
        mid_lat = mid_lat + (mask_lat * 2 - 1) * (180/(1<<p))
        mid_lon = mid_lon + (mask_lon * 2 - 1) * (360/(1<<p))

        GeoHash._encode(lat, lon, mid_lat, mid_lon, pre-1, p+1,hash)
    @staticmethod
    def _d2c(hash):
        hash = hash.copy()
        str_hash = np.full(hash.shape[0],'',dtype=object)
        while (hash>0).any():
            hash1 = list(hash%32)
            str_hash = _base32[hash1] + str_hash
            hash = hash//32
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

        hash = np.zeros(se.shape[0]//2,dtype=np.int64)
        mid_lat = np.full(se.shape[0]//2,90)
        mid_lon = np.full(se.shape[0]//2,180)
        GeoHash._encode(se_lat,se_lon,mid_lat,mid_lon,pre*5//2-1,2,hash)
        if return_str:
            return hash,GeoHash._d2c(hash)
        else:
            return hash
