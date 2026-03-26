import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
from torchvision.datasets import CIFAR10, CIFAR100, CocoDetection
import torch
import os
import sys
from pathlib import Path
import numpy as np
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__name__).resolve().parent))
from torchvision.transforms import InterpolationMode
from torchcp.utils.common import get_device
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset

from CONFIG import DATASET_MAPPINGS

transform_imagenet_resnet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225])
])

transform_imagenet_resnet_v2 = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225])
])

transform_imagenet_vit_L_16 = transforms.Compose([
    transforms.Resize(242),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225])
])

transform_imagenet_vit_B_16_e2e_v1 = transforms.Compose([
    transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225])
])

transform_imagenet_vit_H_14_e2e_v1 = transforms.Compose([
    transforms.Resize(518, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225])
])

transform_imagenet_vit_L_16_e2e_v1 = transforms.Compose([
    transforms.Resize(512, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225])
])

transform_imagenet_linear_v1 = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std= [0.229, 0.224, 0.225])
])

transform_cifar10_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
])

transform_cifar100_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2673, 0.2564, 0.2762))
])

transform_imagenet_efficientnet_b7 = transforms.Compose([
    transforms.Resize(600, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(600),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_imagenet_regnet = transforms.Compose([
    transforms.Resize(384, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_imagenet_resnext = transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_imagenet_convnext = transforms.Compose([
    transforms.Resize(230),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_imagenet_efficientnet_v2 = transforms.Compose([
    transforms.Resize(480, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(480),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_coco = transforms.Compose([           
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

TRANSFORM_MAPPING = {
    'imagenet': {
        'alexnet': {'DEFAULT': transform_imagenet_resnet},
        'resnet18': {'DEFAULT': transform_imagenet_resnet},
        'resnet34': {'DEFAULT': transform_imagenet_resnet},
        'resnet50': {'DEFAULT': transform_imagenet_resnet, 'IMAGENET1K_V2': transform_imagenet_resnet_v2},
        'resnet101': {'DEFAULT': transform_imagenet_resnet, 'IMAGENET1K_V2': transform_imagenet_resnet_v2},
        'resnet152': {'DEFAULT': transform_imagenet_resnet, 'IMAGENET1K_V2': transform_imagenet_resnet_v2,},
        'vit_b_16': {
            'DEFAULT': transform_imagenet_resnet,
            'IMAGENET1K_SWAG_E2E_V1': transform_imagenet_vit_B_16_e2e_v1,
            'IMAGENET1K_SWAG_LINEAR_V1': transform_imagenet_linear_v1,
        },
        'vit_b_32': {
            'DEFAULT': transform_imagenet_resnet,
        },
        'vit_h_14': {
            'DEFAULT': transform_imagenet_vit_H_14_e2e_v1,
            'IMAGENET1K_SWAG_LINEAR_V1': transform_imagenet_linear_v1,
        },
        'vit_l_16': {
            'DEFAULT': transform_imagenet_vit_L_16,
            'IMAGENET1K_SWAG_E2E_V1': transform_imagenet_vit_L_16_e2e_v1,
            'IMAGENET1K_SWAG_LINEAR_V1': transform_imagenet_linear_v1,
        },
        'vit_l_32': {
            'DEFAULT': transform_imagenet_resnet,
        },
        'densenet161': {
            'DEFAULT': transform_imagenet_resnet,
        },
        'densenet121': {
            'DEFAULT': transform_imagenet_resnet,
        },
        'efficientnet_b0': {
            'DEFAULT': transform_imagenet_resnet,
        },
        'efficientnet_b7': {
            'DEFAULT': transform_imagenet_efficientnet_b7,
        },
        'mobilenet_v3_large': {
            'DEFAULT': transform_imagenet_resnet,
        },
        'regnet_y_128gf': {
            'DEFAULT': transform_imagenet_regnet,
        },
        'resnext101_64x4d': {
            'DEFAULT': transform_imagenet_resnext,
        },
        'wide_resnet101_2': {
            'DEFAULT': transform_imagenet_resnet,
        },
        'convnext_small': {
            'DEFAULT': transform_imagenet_convnext,
        },
        'efficientnet_v2_l': {
            'DEFAULT': transform_imagenet_efficientnet_v2,
        },
        'mnasnet0_5': {
            'DEFAULT': transform_imagenet_resnet,
        },
        'shufflenet_v2_x0_5': {
            'DEFAULT': transform_imagenet_resnet,
        }
    },
    'coco': {
        'DEFAULT': transform_coco,
    }
}


def get_class_target_mappings(datadir, folder_name_transform=lambda x: x):
    datadir = Path(datadir)
    classes_sorted = sorted([folder_name_transform(s.name)
                            for s in datadir.glob('*') if s.is_dir()])
    return {k: i for i, k in enumerate(classes_sorted)}


def get_dataset(dataset_name, model_name, weight='DEFAULT'):
    dataset_map = DATASET_MAPPINGS
        
    
    if weight == 'train':
        train = True
        weight = 'DEFAULT'
    else:
        train = False

    if dataset_name == 'imagenet' or dataset_name == 'imagenet-s':
        dataset_dir = dataset_map["imagenet"]
        transform = TRANSFORM_MAPPING["imagenet"][model_name][weight]
        dataset = datasets.ImageFolder(dataset_dir, transform)
    elif dataset_name == 'cifar10':
        dataset_dir = dataset_map["cifar10"]
        transform = transform_cifar10_test
        dataset = CIFAR10(root=dataset_dir, train=train, transform=transform)
    elif dataset_name == 'cifar100':
        dataset_dir = dataset_map["cifar100"]
        transform = transform_cifar100_test
        dataset = CIFAR100(root=dataset_dir, train=train, transform=transform)
    elif dataset_name == 'imagenetv2':
        dataset_dir = dataset_map["imagenetv2"]
        transform = TRANSFORM_MAPPING['imagenet'][model_name][weight]
        dataset = datasets.ImageFolder(dataset_dir, transform)
        new_class_mappings = get_class_target_mappings(dataset_dir)
        new_target_mappings = {v: k for k, v in new_class_mappings.items()}
        def target_transform(x): return int(new_target_mappings[x])
        dataset = datasets.ImageFolder(dataset_dir, transform, target_transform=target_transform)
    elif dataset_name == 'imagenet-r' or dataset_name == 'imagenet-a':
        dataset_dir = dataset_map[dataset_name]
        transform = TRANSFORM_MAPPING['imagenet'][model_name][weight]
        base_class_mappings = get_class_target_mappings(dataset_map['imagenet'])
        new_class_mappings = get_class_target_mappings(dataset_dir)
        new_target_mappings = {v: k for k, v in new_class_mappings.items()}
        def target_transform(x): return base_class_mappings[new_target_mappings[x]]
        dataset = datasets.ImageFolder(dataset_dir, transform, target_transform=target_transform)
    elif dataset_name == 'coco':
        image_dir = dataset_map[dataset_name]['image']
        annotation_file = dataset_map[dataset_name]['annotation']
        transform = TRANSFORM_MAPPING['coco'][weight]
        dataset = CocoDetection(root=image_dir, annFile=annotation_file, transform=transform)
        
    return dataset


def get_device(model):
    return next(model.parameters()).device


def load_or_compute_tensor(file_path, compute_fn, dataloader, device, model, model_name):
    if os.path.isfile(file_path):
        print(f"Loading precomputed tensor from {file_path}")
        return torch.load(file_path)
    else:
        print(f"Precomputed tensor not found. Calculating and saving to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tensor = compute_fn(dataloader, device, model, model_name)
        torch.save(tensor, file_path)
        return tensor

def compute_logits(dataloader, device, model, model_name):
    logits_list = []
    with torch.no_grad():
        for examples in tqdm(dataloader, desc="Computing logits"):
            tmp_x, _ = examples[0].to(device), examples[1].to(device)
            tmp_logits = model(tmp_x)
            logits_list.append(tmp_logits)
            del tmp_logits, tmp_x
            torch.cuda.empty_cache()
    return torch.cat(logits_list).float()


def compute_labels(dataloader, device, model, model_name):
    labels_list = []
    with torch.no_grad():
        for examples in tqdm(dataloader, desc="Computing labels"):
            _, tmp_label = examples[0].to(device), examples[1].to(device)
            labels_list.append(tmp_label)
            del tmp_label
            torch.cuda.empty_cache()
    return torch.cat(labels_list)


def compute_features(dataloader, device, model, model_name):
    if 'vit' in model_name.lower():
        feature_layer = model.encoder
    else:
        feature_layer = nn.Sequential(*list(model.children())[:-1])
    
    features_list = []
    with torch.no_grad():
        for examples in tqdm(dataloader, desc="Computing features"):
            tmp_x = examples[0].to(device)
            if 'vit' in model_name.lower():
                patches = model._process_input(tmp_x)
                n = patches.shape[0]
                batch_class_token = model.class_token.expand(n, -1, -1)
                patches = torch.cat([batch_class_token, patches], dim=1)
                tmp_features = feature_layer(patches)[:, 0]  
            else:
                tmp_features = feature_layer(tmp_x)
                tmp_features = tmp_features.squeeze() 
            
            features_list.append(tmp_features.cpu())
    
    features = torch.cat(features_list).float()
    return features.to(device)


def collate_fn(batch):
    """
    batch: List of (image, target)
    """
    images, targets = list(zip(*batch))  # images: tuple, targets: tuple
    images = torch.stack(images, 0)
    return images, list(targets) 


def get_data_tensor(model, dataset, dataset_name, model_name, file_dir='.tensor', weight='DEFAULT',
                    batch_size=128, num_workers=4, feature_extractor=False):
    logits_path = f'{file_dir}/{dataset_name}_{model_name}_{weight}_logits.pt'
    labels_path = f'{file_dir}/{dataset_name}_{model_name}_{weight}_labels.pt'
    features_path = f'{file_dir}/{dataset_name}_{model_name}_{weight}_features.pt' if feature_extractor else None
    
    device = get_device(model)
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                            num_workers=num_workers, pin_memory=True)
    logits = load_or_compute_tensor(logits_path, compute_logits, dataloader, device, model, model_name).to(device)
    labels = load_or_compute_tensor(labels_path, compute_labels, dataloader, device, model, model_name).to(device)
    features = load_or_compute_tensor(features_path, compute_features, dataloader, device, model, model_name).to(device) if feature_extractor else None

    return (logits, labels, features.squeeze(-1).squeeze(-1)) if feature_extractor else (logits, labels)



def devide_semi_data(logits, labels, labeled_num, unlabeled_num, test_num, features=None):
    """
    Divides data (logits, labels, and optional features) into labeled, unlabeled, and test sets.
    This function is now compatible with both standard Tensors/arrays and lists of Tensors
    (common in object detection).

    Args:
        logits: A Tensor, NumPy array, or a list of Tensors.
        labels: A Tensor, NumPy array, or a list of Tensors.
        labeled_num (int): The number of samples for the labeled set.
        unlabeled_num (int): The number of samples for the unlabeled set.
        test_num (int): The number of samples for the test set.
        features: (Optional) A Tensor, NumPy array, or list corresponding to features.

    Returns:
        A tuple containing the divided data splits.
    """
    total_num = len(logits)
    
    is_list_input = isinstance(logits, list)
    if is_list_input:
        logits_arr = np.array(logits, dtype=object)
        labels_arr = np.array(labels, dtype=object)
        if features is not None:
            features_arr = np.array(features, dtype=object)
    else:
        logits_arr = logits
        labels_arr = labels
        if features is not None:
            features_arr = features
    # -----------------------------

    indices = np.random.permutation(total_num)  

    unlabeled_indices = indices[:unlabeled_num]
    test_indices = indices[unlabeled_num:unlabeled_num + test_num]

    unlabeled_logits = logits_arr[unlabeled_indices]
    unlabeled_labels = labels_arr[unlabeled_indices]
    test_logits = logits_arr[test_indices]
    test_labels = labels_arr[test_indices]
    
    remaining_indices = indices[unlabeled_num + test_num:]

    if len(remaining_indices) >= labeled_num:
        labeled_indices = remaining_indices[:labeled_num]
    else:
        print(f"Warning: Not enough remaining data for the labeled set ({len(remaining_indices)} < {labeled_num}). "
              "Resampling with replacement from the remaining data.")
        labeled_indices = np.random.choice(remaining_indices, size=labeled_num, replace=True)

    labeled_logits = logits_arr[labeled_indices]
    labeled_labels = labels_arr[labeled_indices]

    if features is not None:
        if len(features) != total_num:
            raise ValueError("Error: The length of features does not match logits and labels.")
        unlabeled_features = features_arr[unlabeled_indices]
        test_features = features_arr[test_indices]
        labeled_features = features_arr[labeled_indices]
        
        if is_list_input:
            return (list(labeled_logits), list(labeled_labels), list(labeled_features), 
                    list(unlabeled_logits), list(unlabeled_labels), list(unlabeled_features), 
                    list(test_logits), list(test_labels), list(test_features))
        else:
            return (labeled_logits, labeled_labels, labeled_features, 
                    unlabeled_logits, unlabeled_labels, unlabeled_features, 
                    test_logits, test_labels, test_features)

    if is_list_input:
        return (list(labeled_logits), list(labeled_labels), 
                list(unlabeled_logits), list(unlabeled_labels), 
                list(test_logits), list(test_labels))
    else:
        return (labeled_logits, labeled_labels, 
                unlabeled_logits, unlabeled_labels, 
                test_logits, test_labels)
