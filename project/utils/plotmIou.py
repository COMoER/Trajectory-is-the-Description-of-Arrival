import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import sys
import os
sys.path.append(str(Path(__file__).resolve().parent.parent))
install_path = str(Path(__file__).resolve().parent.parent)

def smooth_curve(y, smooth):
    r = smooth
    length = int(np.prod(y.shape))
    for i in range(length):
        if i > 0:
            if (not np.isinf(y[i - 1])) and (not np.isnan(y[i - 1])):
                y[i] = y[i - 1] * r + y[i] * (1 - r)
    return y


def moving_average(y, x=None, total_steps=1000, smooth=0.3, move_max=False):
    if isinstance(y, list):
        y = np.array(y)
    length = int(np.prod(y.shape))
    if x is None:
        x = list(range(1, length + 1))
    if isinstance(x, list):
        x = np.array(x)
    if length > total_steps:
        block_size = length // total_steps
        select_list = list(range(0, length, block_size))
        select_list = select_list[:-1]
        y = y[:len(select_list) * block_size].reshape(-1, block_size)
        if move_max:
            y = np.max(y, -1)
        else:
            y = np.mean(y, -1)
        x = x[select_list]
    y = smooth_curve(y, smooth)
    return y, x
def get(dirname,item_title):
    f = open(os.path.join(dirname,"log_train.txt"),'r')
    lines = f.readlines()
    f.close()
    data_set = {}
    for i in item_title:
        data_set[i]=[]
    for l in lines:
        for i in item_title:
            if i in l:
                data_set[i].append(float(l.split(':')[-1]))
    return data_set
if __name__ == '__main__':
    smooth = 0.1
    exp = os.path.join(install_path,"log","20220110_160949")
    plt.title("Experiment Result")
    item_title = ['train','eval']
    dataset = get(exp,item_title)
    for i in item_title:
        dataset[i],x = moving_average(np.array(dataset[i]),np.arange(len(dataset[i])),len(dataset[i]),smooth)
        plt.plot(x,dataset[i])
    # plt.ylim([2,3])
    plt.legend(item_title)
    plt.show()

