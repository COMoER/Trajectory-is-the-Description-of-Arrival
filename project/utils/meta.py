from pathlib import Path
import sys
sys.path.append(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
import pickle as pkl
import datetime



def mapMeta(df_train):
    """
    return with extra meta data ['TAXI_ID',"ORIGIN_CALL","ORIGIN_STAND",'day', 'qh', 'qw']
    """
    with open("pretrained/meta_map.pt", 'rb') as f:
        map_dict = pkl.load(f)
    taxi_map, call_map, stand_map = map_dict.values()
    df_train['TAXI_ID'] = [taxi_map[t] for t in df_train['TAXI_ID']]
    df_train["ORIGIN_CALL"] = [call_map[t] for t in df_train["ORIGIN_CALL"].fillna(-1)]
    df_train["ORIGIN_STAND"] = [stand_map[t] for t in df_train["ORIGIN_STAND"].fillna(-1)]
    times = df_train["TIMESTAMP"]
    x = [datetime.datetime.fromtimestamp(time, datetime.timezone.utc) for time in times]
    df_train = df_train.reset_index()
    day = pd.DataFrame(
        np.array([(int(t.strftime("%j")) - 1, (t.minute // 15 + t.hour * 4), (t.weekday()) // 2) for t in x],
                 dtype=int),
        columns=['day', 'qh', 'qw'], dtype='object')
    meta_index = ['TAXI_ID',"ORIGIN_CALL","ORIGIN_STAND",'day', 'qh', 'qw']
    return pd.concat([df_train, day.reset_index()], axis=1),meta_index

def getMetaMap(df_train):
    taxi_id = set(df_train['TAXI_ID'])
    taxi_map = dict(zip(taxi_id, range(len(taxi_id))))

    x = pd.Categorical.describe(df_train["ORIGIN_CALL"].fillna(-1))
    call = x.index.to_list()
    call_map = dict(zip(call, range(len(call))))

    x = pd.Categorical.describe(df_train["ORIGIN_STAND"].fillna(-1))
    stand = x.index.to_list()
    stand_map = dict(zip(stand, range(len(call))))

    with open("pretrained/meta_map.pt", 'wb') as f:
        pkl.dump({'taxi': taxi_map, 'call': call_map, 'stand': stand_map}, f)
