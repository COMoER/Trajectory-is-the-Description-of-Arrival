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

