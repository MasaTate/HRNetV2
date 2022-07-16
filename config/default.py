from yacs.config import CfgNode as CN

_C = CN()

# CUDA related prarams
_C.CUDA = CN()
_C.CUDA.USE_CUDA = True
_C.CUDA.CUDA_NUM = [2,3,8,9]

# dataset related params
_C.DATASET = CN()

_C.DATASET.ROOT = "/work/masatate/dataset/Cityscapes"
_C.DATASET.NUM_CLASSES = 19

# training
_C.TRAIN = CN()

_C.TRAIN.EPOCH_START = 0
_C.TRAIN.EPOCH_END = 480
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.LERNING_RATE = 0.01
_C.TRAIN.NUM_WORKERS = 2
_C.TRAIN.LOG_PATH = './logs'
_C.TRAIN.LOG_LOSS = 50
_C.TRAIN.LOG_IMAGE = 500
_C.TRAIN.SAVE_WEIGHT_PATH = './checkpoint'
_C.TRAIN.SAVE_WEIGHT_STEP = 3000000

_C.TRAIN.CHECKPOINT = ''

# testing
_C.TEST = CN()

_C.TEST.BATCH_SIZE = 1
_C.TEST.NUM_WORKERS = 2
_C.TEST.CHECKPOINT = './checkpoint/checkpoint_epoch40_final.pth'
_C.TEST.RESULTS_NUM = 3
_C.TEST.RESULTS_PATH = './results'


def get_cfg_defaults():
    return _C.clone()