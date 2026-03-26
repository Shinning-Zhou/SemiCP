import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from torchvision.datasets import CIFAR10, CIFAR100
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from .resnet_e5 import resnet50
import CONFIG


def get_model(model_name, dataset_name, device=None, weight='DEFAULT'):
    if dataset_name.lower() == 'imagenet':
        model = get_imagenet_model(model_name, device, weight)
    elif dataset_name.lower() in ['cifar10', 'cifar100']:
        # model = get_cifar_model(model_name, dataset_name, device) 
        model = get_cifar_model(model_name, dataset_name, device, weight)
    else:
        raise ValueError("NOT SUPPORT DATASET")
    if device is not None:  
        model = model.to(device)
            
    return model

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762)),
    transforms.RandomErasing(p=0.25,
                            scale=(0.0625, 0.1),
                            ratio=(0.99, 1.0))
])

def get_cifar_model(model_name, dataset_name, device, weight):
    if dataset_name == 'cifar100':
        model = resnet50(num_classes = 100)
        save_path = CONFIG.PRETRAIN_MODEL_FOR_CIFAR100
        checkpoint = torch.load(save_path, weights_only=False)['model']
        model.load_state_dict(checkpoint)
    elif dataset_name == 'cifar10': 
        model = resnet50(num_classes = 10)
        save_path = CONFIG.PRETRAIN_MODEL_FOR_CIFAR10
        checkpoint = torch.load(save_path, weights_only=False)['model']
        model.load_state_dict(checkpoint)
        
    model = model.to(device)
    model.eval()
    return model


def get_imagenet_model(model_name, device, weight='DEFAULT'):
    model = torch.hub.load("pytorch/vision", model_name, weights=weight, trust_repo=True)
    model.eval()
    model.to(device)
    return model

