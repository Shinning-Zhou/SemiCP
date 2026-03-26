import torch
import numpy as np

from torchcp.classification.utils.metrics import Metrics


@torch.no_grad()
def eval_per_class(prediction_sets: torch.Tensor, test_labels: torch.Tensor, class_num: int):
    # 预测集大小
    set_size = prediction_sets.sum(dim=1).float()  # [N]

    # 覆盖指示：y in set
    idx = torch.arange(test_labels.size(0), device=test_labels.device)
    covered = prediction_sets[idx, test_labels].float()  # [N]

    per_class_coverage = torch.zeros(class_num, device=test_labels.device)
    per_class_avg_size = torch.zeros(class_num, device=test_labels.device)
    per_class_count = torch.zeros(class_num, device=test_labels.device)

    for c in range(class_num):
        mask = (test_labels == c)
        cnt = mask.sum().float()
        per_class_count[c] = cnt
        if cnt > 0:
            per_class_coverage[c] = covered[mask].mean()
            per_class_avg_size[c] = set_size[mask].mean()
        else:
            per_class_coverage[c] = float('nan')
            per_class_avg_size[c] = float('nan')

    overall_coverage = covered.mean().item()
    overall_avg_size = set_size.mean().item()

    return per_class_coverage, per_class_avg_size, per_class_count, overall_coverage, overall_avg_size


@torch.no_grad()
def top1_acc_per_class(logits: torch.Tensor, labels: torch.Tensor, class_num: int):
    """
    logits: [N,K], labels:[N]
    返回 per_class_acc:[K], per_class_count:[K], overall_acc:float
    """
    pred = logits.argmax(dim=1)
    correct = (pred == labels).float()

    per_class_acc = torch.zeros(class_num, device=labels.device)
    per_class_count = torch.zeros(class_num, device=labels.device)

    for c in range(class_num):
        mask = (labels == c)
        cnt = mask.sum().float()
        per_class_count[c] = cnt
        if cnt > 0:
            per_class_acc[c] = correct[mask].mean()
        else:
            per_class_acc[c] = float('nan')

    overall_acc = correct.mean().item()
    return per_class_acc, per_class_count, overall_acc


def group_covgap(prediction_sets, labels, groups, alpha, group_num):
    if prediction_sets.shape[0] != len(labels) or len(labels) != len(groups):
        raise ValueError("Number of prediction sets, labels, and groups must match")
    
    labels = labels.cpu()
    prediction_sets = prediction_sets.cpu()
    groups = groups.cpu()
    
    # 计算每个样本是否被正确覆盖
    covered = prediction_sets[torch.arange(len(labels)), labels]
    
    # 计算每个 group 的样本数量
    group_counts = torch.bincount(groups, minlength=group_num)
    
    # 计算每个 group 的正确覆盖数量
    group_covered = torch.bincount(groups[covered == 1], minlength=group_num)
    
    # 计算每个 group 的覆盖率
    group_coverage_rate = torch.zeros_like(group_counts, dtype=torch.float32)
    valid_groups = group_counts > 0  # 仅计算存在的 group
    group_coverage_rate[valid_groups] = group_covered[valid_groups].float() / group_counts[valid_groups].float()
    
    # 计算每个 group 的 coverage gap
    group_covgap = torch.mean(torch.abs(group_coverage_rate[valid_groups] - (1 - alpha)))
    return group_covgap.float().item()

def class_covgap(prediction_sets, labels, alpha, num_classes):
    if prediction_sets.shape[0] != len(labels):
        raise ValueError("Number of prediction sets must match number of labels")

    labels = labels.cpu()
    prediction_sets = prediction_sets.cpu()

    covered = prediction_sets[torch.arange(len(labels)), labels]
    class_counts = torch.bincount(labels, minlength=num_classes)
    class_covered = torch.bincount(labels[covered==1], minlength=num_classes)

    cls_coverage_rate = torch.zeros_like(class_counts, dtype=torch.float32)
    valid_classes = class_counts > 0
    cls_coverage_rate[valid_classes] = class_covered[valid_classes].float() / class_counts[valid_classes].float()

    overall_covgap = torch.mean(torch.abs(cls_coverage_rate[valid_classes] - (1 - alpha)))
    return overall_covgap.float().item()


def evaluation(setting, prediction_sets, test_labels, alpha, class_num=None, group_num=None, groups=None): # group_num or class_num
    metrics = Metrics()
    coverage_rate = metrics('coverage_rate')(prediction_sets, test_labels)
    average_size = metrics('average_size')(prediction_sets, test_labels)
    if type(average_size) == torch.Tensor:
        average_size = average_size.item()
    if setting == 'margin':
        CovGap = np.abs(coverage_rate - (1 - alpha))
    elif setting == 'class_conditional':
        CovGap = class_covgap(prediction_sets, test_labels, alpha, class_num)
    elif setting == 'group_conditional':
        CovGap = group_covgap(prediction_sets, test_labels, groups, alpha, group_num)
    else:
        raise ValueError("NOT SUPPORT SETTING!")
    
    print(f"Coverage_rate: {coverage_rate}, Average_size: {average_size}, CovGap: {CovGap}")
    return coverage_rate, average_size, CovGap


def top_logit_acc(logits, labels, top=1):

    with torch.no_grad(): #确保不计算梯度
        batch_size = labels.size(0)
        _, pred = logits.topk(top, dim=1, largest=True, sorted=True) #获取topk的预测结果和索引
        pred = pred.t() #转置，方便后续比较
        correct = pred.eq(labels.view(1, -1).expand_as(pred)) #比较预测结果和真实标签
        correct_k = correct[:top].reshape(-1).float().sum(0, keepdim=True) #计算正确的个数
        return correct_k.mul_(100.0 / batch_size).item() #计算准确率并返回
