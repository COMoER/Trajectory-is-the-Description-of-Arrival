### 时空数据挖掘

> 第二十三组 郑煜 刘昊 李嘉霖

##### 课题简介

我们大作业项目将基于Kaggle [ECML/PKDD 15: Taxi Trajectory Prediction (I)]([ECML/PKDD 15: Taxi Trajectory Prediction (I) | Kaggle](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i)) Competition，将对数据集中数据进行分析，根据时空数据的特性，对中间的位置序列以及时间进行数据预处理，提取出相应的discriminative feature，最后采用回归模型得到预测结果

##### 目前进展

- 确定数据
- 对数据集特征进行了组内讨论
- 阅读了[综述]([Spatio-Temporal Data Mining: A Survey of Problems and Methods: ACM Computing Surveys: Vol 51, No 4](https://dl.acm.org/doi/abs/10.1145/3161602)),调研了时空数据的特征以及了解一些处理序列以及时间数据的算法
- 初步确定方法
    - 将基于[shapelets]([基于Shapelet的时间序列分类方法实战 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/359666547))对spatial trajectory提取discriminative feature
    - 将对连续的经纬度信息进行聚类以便上述操作
    - 分析有效具有周期性的时间特征