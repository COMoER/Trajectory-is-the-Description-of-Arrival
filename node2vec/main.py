import numpy as np
from algo import deepwork
from pathlib import Path
import sys
parent = Path(__file__).resolve().parent
sys.path.append(parent)
from datetime import datetime
import os
import argparse
import pandas as pd
import re
import pickle as pkl
from utils import LonLatVocal
if __name__ == '__main__':
    ###### param #########
    parser = argparse.ArgumentParser()

    parser.add_argument("--dimension",type = int,default=32,help="dimension of node vector")
    # epochs, in my training, when reaching 15 epoch, the loss delays slowly, evenly delay the learning rate
    # so 15 epoch is enough
    parser.add_argument("--gamma", type=int, default=15, help="epoch of training")
    parser.add_argument("--K", type=int, default=10, help="neg sampling length")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--last_train_dir", type = str,default="run_2021-12-02_09_34_47", help="learning/testing")
    parser.add_argument("--weight", type=int, default=15, help="mush %5 == 0, the epoch of saved weights")
    #######################
    args = parser.parse_args()
    for name,val in vars(args).items():
        print("%s: %s"%(name,str(val)))
    last_train_dir = Path(args.last_train_dir)


    # load the dataset
    print("[INFO] loading dataset")
    with open(str(parent.parent/Path("geohashmodel.pt")), 'rb') as f:
        model_geo = pkl.load(f)
    model = deepwork(args.dimension,len(model_geo)+1,args.gamma,args.K,args.lr)
    sample_df = pd.read_csv(str(parent.parent/Path("sample_data.csv")))
    texts = list(map(lambda x: list(map(lambda y: int(y), re.findall(r"\d+", x))), sample_df["seq"]))

    arrival = sample_df['arrival'].to_list()
    model.load_data(texts,arrival)

    print("[INFO] finish loading")
    save_dir = Path(__file__).resolve().parent / Path("run_%s" % (datetime.now().strftime("%Y-%m-%d_%H_%M_%S")))
    os.mkdir(save_dir)
    # save_dir = Path("run_result")
    model.train(save_dir)


