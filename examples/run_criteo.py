import sys

sys.path.append("../")

import numpy as np
import pandas as pd
import torch
import argparse
from torch_rechub.models.ranking import WideDeep, DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic import cfg, update_config
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.basic.utils import DataGenerator, set_seed
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train model for criteo dataset")

    # general
    parser.add_argument("--cfg",
                        help="experiment configure file name",
                        # required=True,
                        type=str)
    parser.add_argument('--dataset_path',
                        help="the training dataset path",
                        default="./data/criteo/criteo_sample.csv")
    parser.add_argument('--model_name',
                        help="training model name",
                        default='widedeep')
    parser.add_argument('--epoch',
                        help="training epochs",
                        type=int,
                        default=2)  # 100
    parser.add_argument('--learning_rate',
                        help="learning rate for optimizers",
                        type=float,
                        default=1e-3)
    parser.add_argument('--batch_size',
                        help="size of each training bach",
                        type=int,
                        default=16)  # 4096
    parser.add_argument('--weight_decay',
                        help="weight of decay",
                        type=float,
                        default=1e-3)
    parser.add_argument('--device',
                        help="used device for training, either cpu or cuda:0",
                        default='cpu')  # cuda:0
    parser.add_argument('--save_dir',
                        help="model save directory",
                        default='./')
    parser.add_argument('--seed',
                        help="seed for everything",
                        type=int,
                        default=2022)

    args = parser.parse_args()

    return args


def convert_numeric_feature(val):
    v = int(val)
    if v > 2:
        return int(np.log(v) ** 2)
    else:
        return v - 2


def get_criteo_data_dict(data_path):
    data = pd.read_csv(data_path)
    dense_features = [f for f in data.columns.tolist() if f[0] == "I"]
    sparse_features = [f for f in data.columns.tolist() if f[0] == "C"]

    data[sparse_features] = data[sparse_features].fillna('-996', )
    data[dense_features] = data[dense_features].fillna(0, )

    for feat in tqdm(dense_features):  # discretize dense feature and as new sparse feature
        sparse_features.append(feat + "_cat")
        data[feat + "_cat"] = data[feat].apply(lambda x: convert_numeric_feature(x))

    sca = MinMaxScaler()  # scaler dense feature
    data[dense_features] = sca.fit_transform(data[dense_features])

    for feat in tqdm(sparse_features):  # encode sparse feature
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(feature_name) for feature_name in dense_features]
    sparse_feas = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=16) for feature_name
                   in sparse_features]
    y = data["label"]
    del data["label"]
    x = data
    return dense_feas, sparse_feas, x, y


if __name__ == '__main__':
    args = parse_args()
    args.cfg = '../experiments/criteo/deepfm/deepfm_adam_lr-le-2_ep-10_bs-1e4.yaml'
    update_config(cfg, args)
    set_seed(args.seed)
    dense_feas, sparse_feas, x, y = get_criteo_data_dict(cfg.DATASET.TRAIN_SET)
    dg = DataGenerator(x, y)
    train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.8, 0.1],
                                                                               batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU)
    if cfg.MODEL.NAME == "widedeep":
        model = WideDeep(wide_features=dense_feas, deep_features=sparse_feas,
                         mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})
    elif cfg.MODEL.NAME == "deepfm":
        model = DeepFM(deep_features=dense_feas, fm_features=sparse_feas,
                       mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})

    ctr_trainer = CTRTrainer(model, optimizer_params={"lr": cfg.TRAIN.LR, "weight_decay": cfg.TRAIN.WD},
                             n_epoch=cfg.TRAIN.END_EPOCH, earlystop_patience=4, device=cfg.DEVICE,
                             model_path=cfg.OUTPUT_DIR)
    # scheduler_fn=torch.optim.lr_scheduler.StepLR,scheduler_params={"step_size": 2,"gamma": 0.8},
    ctr_trainer.fit(train_dataloader, val_dataloader)
    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
    print(f'test auc: {auc}')
"""
python run_criteo.py --model_name widedeep
python run_criteo.py --model_name deepfm
"""
