from .data import *
from .model import get_model
from .evaluation import evaluation, top_logit_acc, top1_acc_per_class, eval_per_class
from .group_utils import cluster_feature, get_groups
from .log_utils import filenames, write_result_semicp, write_per_class_csv

import torch
import random
import numpy as np

def fix_randomness(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # For GPU
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.random.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensures deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disables the auto-tuner to ensure deterministic results