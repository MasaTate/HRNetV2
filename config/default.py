from yacs.config import CfgNode as CN

_C = CN()

# CUDA related prarams
_C.CUDA = CN()
_C.CUDA.USE_CUDA = True
_C.CUDA.CUDA_NUM = 0

# dataset related params
_C.DATASET = CN()

_C.DATASET.ROOT = "/work/masatate/dataset/Cityscapes"
_C.DATASET.NUM_CLASSES = 19

# training
_C.TRAIN = CN()

_C.TRAIN.EPOCH_START = 0
_C.TRAIN.EPOCH_END = 10
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.NUM_WORKERS = 2
_C.TRAIN.LOG_PATH = './logs'
_C.TRAIN.LOG_LOSS = 10
_C.TRAIN.LOG_IMAGE = 300
_C.TRAIN.SAVE_WEIGHT_PATH = './checkpoint'
_C.TRAIN.SAVE_WEIGHT_STEP = 500

_C.TRAIN.CHECKPOINT = ''


def get_cfg_defaults():
    return _C.clone()