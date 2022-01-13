###  EDA

##### geohash

how to train geohash

```python
import pickle as pkl
model = LonLatVocal()
model.fit(df_train[:100000],n=-1,verbose=True)
with open("geohashmodel.pt",'wb') as f:
    pkl.dump(model,f)
```

```python
def fit(self,df,n=100000,quantile=[0.01,0.99],verbose=False):
    series = str2trajectory(df,n)
    se = np.concatenate(series)
    lat = se[1::2]
    lon = se[::2]
    def getQmask(x):
        return np.logical_and(x<=np.quantile(x,quantile[1]),x>=np.quantile(x,quantile[0]))
    mask = np.logical_and(getQmask(lat),getQmask(lon))
```

- Origin size 9550022
- [0.01,0.99] 
- reserved size 9264260 
- reserved class size 56959

on the given `kaggle` test dataset

![geohash](https://github.com/COMoER/Trajectory-is-the-Description-of-Arrival/img/geohash.png)

##### approximate method

```
error of this method is 0.01068883 km
```

origin

![origin](https://github.com/COMoER/Trajectory-is-the-Description-of-Arrival/img/origin.png)

trans

![trans](https://github.com/COMoER/Trajectory-is-the-Description-of-Arrival/img/trans.png)

##### time period analysis

```python
import datetime
times = sample_df[mask][mask_q]["TIMESTAMP"]
x = [datetime.datetime.fromtimestamp(time, datetime.timezone.utc) for time in times]
day = pd.DataFrame(np.array([(t.hour,t.month,t.weekday(),t.isocalendar()[1],(t.isocalendar()[1]-1)%26,(t.minute//15+t.hour*4),t.minute,t.second,(t.weekday())//2,(t.weekday()//4),int(t.strftime("%j"))-1)for t in x],dtype=int),
                   columns=['hour','month','weekday','week','semimonth','qh','minute','second','qw','sw','day'],dtype='object')
pd.Categorical.describe(day['month'])
```

```python
for i in range(N):
    mdata = y[timehh==i]
    center = np.mean(mdata,0)
    va.append(np.sqrt(np.mean(np.sum((mdata-center)**2,axis=1))))
    mean.append(center)
```

std of std

year scale

| period    | variance    |
| --------- | ----------- |
| month     | 0.018240372 |
| semimonth | 0.01916852  |
| week      | 0.025951218 |
| day       | 0.0459778   |

week scale

| period       | variance    |
| ------------ | ----------- |
| weekday      | 0.023479585 |
| semiweek     | 0.008631825 |
| quarter week | 0.025702417 |

day scale

| period       | variance    |
| ------------ | ----------- |
| hour         | 0.08774563  |
| quarter hour | 0.089424565 |
| minute       | 0.02010388  |
| second       | 0.023705937 |