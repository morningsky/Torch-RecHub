# 召回

## Movielens

使用ml-1m数据集，使用其中原始特征7个user特征`'user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip',"cate_id"`，2个item特征`"movie_id", "cate_id"`，一共9个sparse特征。

- 构造用户观看历史特征``hist_movie_id``，使用`mean`池化该序列embedding
- 使用随机负采样构造负样本
- 将每个用户最后一条观看记录设置为测试集
- 原始数据下载地址：https://grouplens.org/datasets/movielens/1m/
- 处理数据csv下载地址：https://cowtransfer.com/s/5a3ab69ebd314e

| Model\Metrics | Hit@100 | Recall@100 | Precision@100 |
| ------------- | ------- | ---------- | ------------- |
| DSSM          | 2.43%   | 2.43%      | 0.02%         |
|               |         |            |               |
|               |         |            |               |