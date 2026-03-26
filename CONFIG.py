
from datetime import datetime
import pytz
import numpy as np

from src.score import *

# Since the pytorch only provides the official pretrained model of IMAGENET, we provide the pretrained models of CIFAR10 and CIFAR100
# PRETRAIN_MODEL_FOR_CIFAR10 = './pretrain_model/cifar10_resnet50.pth'
# PRETRAIN_MODEL_FOR_CIFAR100 = './pretrain_model/cifar100_resnet50.pth'

DATASET_MAPPINGS = {
    "cifar10": "/path/to/your/cifar10/dataset",
    "cifar100": "/path/to/your/cifar100/dataset",
    "imagenet": "/path/to/your/imagenet/val",
    "imagenetv2": "/path/to/your/imagenetv2",
    "imagenet-r": "/path/to/your/imagenet-r",
    "imagenet-s": "/path/to/your/imagenet-sketch",
    "imagenet-a": "/path/to/your/imagenet-a",
}

# common config
cuda = 0
seeds = range(0, 1000)
batch_size = 128
num_workers = 6
tensor_path = "./tensor"
impute_accs = [None]
group_num = 20

# cp config
# config for fig 4.1
# settings = ["margin"]
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['cifar10', 'resnet50', 'DEFAULT']]
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 80, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [4000]
# test_lens = [2000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig4_1.csv"

# config for fig 4.2
# settings = ["margin"]
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['cifar100', 'resnet50', 'DEFAULT']]
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 80, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [4000]
# test_lens = [2000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig4_2.csv"

# ## config for fig 4.3
settings = ["margin"]
temperature = 1
alphas = [0.1]
random = False
dataset_model_weight = [['imagenet', 'resnet50', 'DEFAULT']]
scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
unlabeled_score = UNS(randomized=random)
avg_labeled_lens = [10, 20, 40, 60, 80, 100]
restrict_len = max(avg_labeled_lens)
unlabeled_lens = [20000]
test_lens = [10000]
folder_path = f"./output/"
result_fname = f"{folder_path}/reproduce_semicp_fig4_3.csv"

# ## config for fig 5
# settings = ["margin"]
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['imagenet', 'resnet50', 'DEFAULT']]
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [50]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 40000]
# test_lens = [5000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig5.csv"

# ## config for fig 6
# settings = ["group_conditional"]
# group_num = 20
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['cifar100', 'resnet50', 'DEFAULT']]
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 30, 40]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [4000]
# test_lens = [2000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig6.csv"

# ## config for fig 7
# settings = ["cp_interpolation", "cp_interpolation"]
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['cifar100', 'resnet50', 'DEFAULT']]
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [4000]
# test_lens = [2000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig7.csv"

# ## config for fig 8
# settings = ["margin"]
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [
#     ['imagenet', 'mobilenet_v3_large', 'DEFAULT'], #74
#     ['imagenet', 'convnext_small', 'DEFAULT'], #83
#     ['imagenet', 'efficientnet_v2_l', 'DEFAULT'], #85.8
#     ['imagenet', 'resnet50', 'DEFAULT'], #80
#     ['imagenet', 'resnet152', 'DEFAULT'], #82
#     ['imagenet', 'resnext101_64x4d', 'DEFAULT'], #82
#     ['imagenet', 'wide_resnet101_2', 'DEFAULT'], #78.8
#     ['imagenet', 'mnasnet0_5', 'DEFAULT'], #67
#     ['imagenet', 'regnet_y_128gf', 'DEFAULT'], #88.2 
#     ['imagenet', 'vit_b_16', 'DEFAULT'], #81
#     ['imagenet', 'vit_h_14', 'IMAGENET1K_SWAG_LINEAR_V1'], #85
#     ['imagenet', 'vit_l_16', 'IMAGENET1K_SWAG_E2E_V1'],  # 88.06
# ] ## all model can be found in https://docs.pytorch.org/vision/main/models.html
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [20000]
# test_lens = [10000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig8.csv"

# ## config for fig 10.1
# settings = ["margin"]
# temperature = 1
# alphas = [0.1]
# random = True
# dataset_model_weight = [['cifar10', 'resnet50', 'DEFAULT']]
# scores = [SAPS(weight=0.01, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 80, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [4000]
# test_lens = [2000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig10_1.csv"

# ## config for fig 10.1
# settings = ["margin"]
# temperature = 1
# alphas = [0.1]
# random = True
# dataset_model_weight = [['cifar100', 'resnet50', 'DEFAULT']]
# scores = [SAPS(weight=0.01, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 80, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [4000]
# test_lens = [2000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig10_2.csv"

# ## config for fig 10.3
# settings = ["margin"]
# temperature = 1
# alphas = [0.1]
# random = True
# dataset_model_weight = [['imagenet', 'resnet50', 'DEFAULT']]
# scores = [SAPS(weight=0.01, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 80, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [20000]
# test_lens = [10000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig10_3.csv"

# ## config for fig 11
# settings = ["impute_acc"]
# mpute_accs = np.arange(0.6, 1.01, 0.02)
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['imagenet', 'resnet50', 'DEFAULT']]
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [20000]
# test_lens = [10000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig11.csv"

# ## config for table3
# settings = ["shift"]
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['imagenet', 'resnet50', 'DEFAULT']]
# shift_dataset_model_weight = ['imagenet-r', 'resnet50', 'DEFAULT'] ## ['imagenet-s', 'resnet50', 'DEFAULT'] /  ['imagenetv2', 'resnet50', 'DEFAULT']
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [20]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [20000]
# test_lens = [10000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_table3.csv"

# ## config for fig 12
# settings = ["ablation_debias"]
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['imagenet', 'resnet50', 'DEFAULT']]
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 80, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [20000]
# test_lens = [10000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig12.csv"

# ## config for fig 13
# settings = ["ablation_nn"]
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['imagenet', 'resnet50', 'DEFAULT']]
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 80, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [20000]
# test_lens = [10000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig13.csv"

# ## config for fig 14
# settings = ["nerghbor_num"]
# temperature = 1
# alphas = [0.1]
# random = False
# dataset_model_weight = [['imagenet', 'resnet50', 'DEFAULT']]
# scores = [THR(), APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 40, 60, 80, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [20000]
# test_lens = [10000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_fig14.csv"

# ## config for table 4
# settings = ["margin"]
# temperature = 1
# alphas = [0.1]
# random = True
# dataset_model_weight = [['imagenet', 'resnet50', 'DEFAULT']] # cifar10/cifar100
# scores = [APS(randomized=random), RAPS(penalty=0.01, kreg=2, randomized=random)]
# unlabeled_score = UNS(randomized=random)
# avg_labeled_lens = [10, 20, 50, 100]
# restrict_len = max(avg_labeled_lens)
# unlabeled_lens = [20000]
# test_lens = [10000]
# folder_path = f"./output/"
# result_fname = f"{folder_path}/reproduce_semicp_table4.csv"