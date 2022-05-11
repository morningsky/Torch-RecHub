import os
from yacs.config import CfgNode as CN

_C = CN()
_C.AUTO_RESUME = False
_C.DEVICE = 'cpu'
_C.GPUS = (0,)
_C.DATA_DIR = ''
_C.OUTPUT_DIR = ''
_C.LOG_DIR = ''
_C.WORKERS = 4
_C.PRINT_FREQ = 100
_C.SEED = 2022

# Cudnn related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# DATASET related params
_C.DATASET = CN()
_C.DATASET.DATASET = 'criteo'
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN_SET = 'train'
_C.DATASET.VAL_SET = 'val'
_C.DATASET.TEST_SET = 'test'

# common params for NETWORK
_C.MODEL = CN()
_C.MODEL.NAME = 'deepfm'
_C.MODEL.SPARSE_EMB_DIM = 8
_C.MODEL.DNN_UNITS = (64, 32)
_C.MODEL.CIN_LAYER_SIZE = (256, 128)
_C.MODEL.TEACHER_NAME = ''
_C.MODEL.TEACHER_SPARSE_EMB_DIM = 32
_C.MODEL.TEACHER_DNN_UNITS = (512, 256, 128, 64)
_C.MODEL.PRETRAINED = ''
_C.MODEL.TEACHER_PRETRAINED = ''
_C.MODEL.GNN_LAYERS = 3
_C.MODEL.REUSE_GRAPH_LAYER = False
_C.MODEL.USE_GRU = False
_C.MODEL.USE_RESIDUAL = False
_C.MODEL.ALPHA = 0.1
_C.MODEL.BETA = 0.9

_C.LOSS = CN()
_C.LOSS.USE_TARGET_WEIGHT = True

# train
_C.TRAIN = CN()
_C.TRAIN.SAVE_MODEL = False
_C.TRAIN.SAVE_MODEL_PREFIX = ''
_C.TRAIN.SAVE_MODEL_POSTFIX = ''
_C.TRAIN.BATCH_SIZE_PER_GPU = 1000
_C.TRAIN.SHUFFLE = True
_C.TRAIN.PIN_MEMORY = False
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 140
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.LR_SCHEDULER = 'reduce_lr_on_plateau'
_C.TRAIN.LR = 0.001
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.LR_STEP = (90, 110)
_C.TRAIN.WD = 0.0001
_C.TRAIN.GAMMA1 = 0.99
_C.TRAIN.GAMMA2 = 0.0
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.NESTEROV = False
_C.TRAIN.L2_REG_LINEAR = 1e-5
_C.TRAIN.L2_REG_EMBEDDING = 1e-5
_C.TRAIN.L2_REG_DNN = 1e-5
_C.TRAIN.DNN_DROPOUT = 0.0

# testing
_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 2000
_C.TEST.MODEL_FILE = ''

# kd params
_C.KD = CN()
_C.KD.TRAIN_TYPE = 'NORMAL'  # NORMAL KD
_C.KD.TEACHER = ''  # teacher model
_C.KD.ALPHA = 0.5  # kd weight
_C.KD.TEMPERATURE = 1.0
_C.KD.LOGIT_LOSS_WEIGHT = 10.
_C.KD.HINT_REGRESSION_WEIGHT = (10.,)
_C.KD.KNOWLEDGE_REGRESSION_WEIGHT = 10.


def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)

    cfg.MODEL.PRETRAINED = os.path.join(
        cfg.DATA_DIR, cfg.MODEL.PRETRAINED
    )

    if cfg.TEST.MODEL_FILE:
        cfg.TEST.MODEL_FILE = os.path.join(
            cfg.DATA_DIR, cfg.TEST.MODEL_FILE
        )

    cfg.freeze()


def modify_config(cfg, key, value):
    cfg.defrost()
    key = value
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
