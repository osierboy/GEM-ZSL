import os

from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"

_C.MODEL.META_ARCHITECTURE = "GEMModel"
_C.MODEL.NAME = ""

_C.MODEL.WEIGHT = ""

_C.MODEL.SCALE = 20.0



# -----------------------------------------------------------------------------
# Backbone, ResNet101
# -----------------------------------------------------------------------------
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.PRETRAINED = True


# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------
_C.MODEL.ATTENTION = CN()
_C.MODEL.ATTENTION.MODE = 'add' # 'add', 'concat'
_C.MODEL.ATTENTION.CHANNEL = 512
_C.MODEL.ATTENTION.WEIGHT_SHARED = True
_C.MODEL.ATTENTION.W2V_PATH = "datasets/Attrbute/w2v"

# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
_C.MODEL.LOSS = CN()
_C.MODEL.LOSS.LAMBDA1 = 1.0
_C.MODEL.LOSS.LAMBDA2 = 0.05
_C.MODEL.LOSS.LAMBDA3 = 0.2
_C.MODEL.LOSS.LAMBDA4 = 0.1

_C.MODEL.LOSS.TEMP = 0.07


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAME = "CUB"
_C.DATASETS.IMAGE_SIZE = 224
_C.DATASETS.WAYS = 16
_C.DATASETS.SHOTS = 4

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.N_BATCH = 1000
_C.DATALOADER.EP_PER_BATCH = 1
_C.DATALOADER.MODE = 'random'  # random, episode


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_EPOCH = 100

_C.SOLVER.BASE_LR = 1e-3
_C.SOLVER.BIAS_LR_FACTOR = 2


_C.SOLVER.WEIGHT_DECAY = 1e-5
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.GAMMA = 0.5
_C.SOLVER.STEPS = 10

_C.SOLVER.CHECKPOINT_PERIOD = 50

_C.SOLVER.DATA_AUG = "resize_random_crop"

_C.SOLVER.RESUME_OPTIM = False
_C.SOLVER.RESUME_SCHED = False

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 100
_C.TEST.DATA_AUG = "resize_crop"
_C.TEST.GAMMA = 0.7


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."
_C.LOG_FILE_NAME = ""
_C.MODEL_FILE_NAME = ""
_C.PRETRAINED_MODELS = "./pretrained_models"


# ---------------------------------------------------------------------------- #
# Precision options
# ---------------------------------------------------------------------------- #

# Precision of input, allowable: (float32, float16)
_C.DTYPE = "float32"
