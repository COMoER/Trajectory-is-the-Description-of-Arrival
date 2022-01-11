import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
def pad(seq):
    seq=seq[:-2]
    lon=seq[::2]
    lat=seq[1::2]
    f=np.vstack((lon,lat)) 
    f=f.T
    if(f.shape[0]<5):
        padding=np.tile(f[-1],(5-f.shape[0],1))
    res=np.vstack((f,padding))
    return res
seq=np.array([1,2,3,4,5,6,7,8])
seq2=np.array([9,10,11,12])
seq3=np.array([13,14,15,16])
s=pad(seq)
s_tensor=torch.Tensor(s)
a,b=s_tensor.shape
s_tensor=s_tensor.view(1,a,b)
print(s_tensor)
print(s_tensor.shape)
'''
s=[]
s.append(pad(seq))
s.append(pad(seq2))
s.append(pad(seq3))
s_tensor=torch.Tensor(s)
print(s_tensor.shape) 
s_tensor=s_tensor.permute(0,2,1)
print(s_tensor.shape)
conv = nn.Conv1d(2,7, kernel_size=3, padding=1)
l1 = F.relu(conv(s_tensor))
print(l1.shape)
l1_max = torch.max(l1, dim=-1)[0]
print(l1_max.shape)
features=torch.randn(3,32)
print(features.shape)
features=torch.cat((l1_max, features), dim=1)
print(features.shape)
'''