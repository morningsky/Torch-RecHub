# Torch-RecHub

<p align="left">
  <img src='https://img.shields.io/badge/python-3.8+-brightgreen'>
  <img src='https://img.shields.io/badge/torch-1.7+-brightgreen'>
  <img src='https://img.shields.io/badge/scikit_learn-0.23.2+-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5+-brightgreen'>
  <img src="https://img.shields.io/pypi/l/torch-rechub">
  <img src='https://img.shields.io/badge/annoy-1.17.0-brightgreen'>
  <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fmorningsky%2FTorch-RecHub%2F&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com">


A Lighting Pytorch Framework for Recommendation Models, Easy-to-use and Easy-to-extend.

## 安装

```python
#稳定版
pip install torch-rechub
#最新版
git clone https://github.com/morningsky/Torch-RecHub.git
```

## 主要特性

- scikit-learn风格易用的API（fit、predict），即插即用
- 训练过程与模型定义解耦，易拓展，可针对不同类型的模型设置不同的训练机制
- 使用Pytorch原生Dataset、DataLoader，易修改，自定义数据
- 高度模块化，支持常见Layer（MLP、FM、FFM、target-attention、self-attention、transformer等），容易调用组装成新模型
- 支持常见排序模型（WideDeep、DeepFM、DIN、DCN、xDeepFM等）

- [ ] 支持常见召回模型（DSSM、YoutubeDNN、MIND、SARSRec等）
- 丰富的多任务学习支持
  - SharedBottom、ESMM、MMOE、PLE、AITM等模型
  - GradNorm、UWL等动态loss加权机制

- 聚焦更生态化的推荐场景
  - [ ] 冷启动
  - [ ] 延迟反馈
  - [ ] 去偏
- [ ] 支持丰富的训练机制（对比学习、蒸馏学习等）

- [ ] 第三方高性能开源Trainer支持（Pytorch Lighting等）
- [ ] 更多模型正在开发中

## 快速使用

```python
from torch_rechub.rmodels.ranking import WideDeep, DeepFM, DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.utils import DataGenerator

dg = DataGenerator(x, y)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader()

model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})

ctr_trainer = CTRTrainer(model)
ctr_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)


```

多任务学习

```python
from torch_rechub.models.multi_task import SharedBottom, ESMM, MMOE, PLE, AITM
from torch_rechub.trainers import MTLTrainer

model = MMOE(features, task_types, n_expert=3, expert_params={"dims": [64,32,16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])

ctr_trainer = MTLTrainer(model)
ctr_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
```





> **Note:** 
>
> 所有模型均在大多数论文提及的多个知名公开数据集中测试，达到或者接近论文性能。
>
> 使用案例：[Examples](./examples)
>
> 每个数据集将会提供
>
> - 一个使用脚本，包含样本生成、模型训练与测试，并提供一套测评所用参数。
> - 一个预处理脚本，将原始数据进行预处理，转化成csv。
> - 数据格式参考文件（100条）。
> - 全量数据，统一的csv文件，提供高速网盘下载链接和原始数据链接。



[初步规划TODO清单](https://user-images.githubusercontent.com/11856746/167436396-f9c5de5b-d341-4697-8b91-884d4ae552be.png)

