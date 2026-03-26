import torch
import torch.nn.functional as F

class UNS():
    "unlabeled nonconformity score"
    def __init__(self, randomized=False):
        self.randomized = randomized
        
    def __call__(self, unlabeled_logits, labeled_logits, labeled_labels, score_function):
        score_function.randomized = False
        labeled_true_score = score_function(logits=labeled_logits, label=labeled_labels)
        labeled_all_score = score_function(logits=labeled_logits)
        unlabeled_all_score = score_function(logits=unlabeled_logits)
        
        _, max_label_logits_idx = torch.kthvalue(labeled_logits, labeled_all_score.size(1), dim=1)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)
        
        max_labeled_score = labeled_all_score[torch.arange(labeled_all_score.size(0)), max_label_logits_idx]
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx]
        
        score_diff = labeled_true_score - max_labeled_score
        closest_indices, closest_score_diff = self.cal_closest_indices(max_labeled_score, max_unlabeled_score, score_diff)

        if (self.randomized == False) or (score_function.__class__.__name__ == 'THR'):
            unlabeled_adjust_score = max_unlabeled_score + closest_score_diff
        else:
            score_function.randomized = True
            U_all = torch.rand(unlabeled_logits.shape, device=labeled_logits.device) #N*K
            U_single = U_all[torch.arange(U_all.size(0)), max_unlabeled_logits_idx] #N*1
            
            unlabeled_all_score_r = score_function(logits=unlabeled_logits, label=None, U=U_all) #N*K
            labeled_logits_all = labeled_logits[closest_indices] #N*K
            labeled_labels_all = labeled_labels[closest_indices] #N*1
            labeled_true_score_all = score_function(logits=labeled_logits_all, label=labeled_labels_all, U=U_single) #N*1
            labeled_all_score_all = score_function(logits=labeled_logits_all, label=None, U=U_all) #N*K
            max_label_logits_idx_all = max_label_logits_idx[closest_indices]
            max_labeled_score_all = labeled_all_score_all[torch.arange(labeled_all_score_all.size(0)), max_label_logits_idx_all]
            max_unlabeled_score_r = unlabeled_all_score_r[torch.arange(unlabeled_all_score_r.size(0)), max_unlabeled_logits_idx]
            score_diff_r = labeled_true_score_all - max_labeled_score_all
            closest_score_diff_r = score_diff_r
            
            unlabeled_adjust_score = max_unlabeled_score_r + closest_score_diff_r
        
        return unlabeled_adjust_score, max_unlabeled_logits_idx
    
    def cal_closest_indices(self, max_labeled_score, max_unlabeled_score, score_diff):
        closest_indices = torch.abs(max_labeled_score[:, None] - max_unlabeled_score[None, :]).argmin(axis=0)
        closest_score_diff = score_diff[closest_indices]
            
        return closest_indices, closest_score_diff

def generate_percentiles(score):
    sorted_scores, sorted_indices = torch.sort(score)
    percentiles = (torch.argsort(sorted_indices).float() + 1) / len(score) * 100
    
    return percentiles

class UNS_conf(UNS):
    """
    Debiased Unlabeled Nonconformity Score based on Confidence (max logit)
    Inherits from UNS but uses max(logits) as the score function.
    """
    def __init__(self, randomized=False):
        super().__init__(randomized=randomized)

    def __call__(self, unlabeled_logits, labeled_logits, labeled_labels=None, score_function=None):
        score_function.randomized = False
        labeled_true_score = score_function(logits=labeled_logits, label=labeled_labels)
        labeled_all_score = score_function(logits=labeled_logits)
        unlabeled_all_score = score_function(logits=unlabeled_logits)
        
        _, max_label_logits_idx = torch.kthvalue(labeled_logits, labeled_all_score.size(1), dim=1)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)
        
        max_labeled_score = labeled_all_score[torch.arange(labeled_all_score.size(0)), max_label_logits_idx]
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx]
        
        # Step 1: compute confidence score as max(logit) over classes
        max_labeled_conf, _ = labeled_logits.max(dim=1)  # (n,)
        max_unlabeled_conf, _ = unlabeled_logits.max(dim=1)  # (m,)

        score_function.randomized = False
        labeled_true_score = score_function(logits=labeled_logits, label=labeled_labels)
        
        score_diff = labeled_true_score - max_labeled_score

        closest_indices, _ = self.cal_closest_indices(max_labeled_conf, max_unlabeled_conf, score_diff)

        # Step 3: debias the unlabeled score using closest labeled score (trivially zero shift here)
        # However, you could model `score_diff = true_class_logit - confidence` if you wish
        if self.randomized is False:
            # debias via matched score_diff (zero here)
            closest_score_diff = score_diff[closest_indices]
            unlabeled_adjust_score = max_unlabeled_score + closest_score_diff
        else:
            # randomized case – perturb with uniform noise
            U = torch.rand_like(unlabeled_logits)
            U_idx = U.argmax(dim=1)
            max_unlabeled_score_r = unlabeled_logits.gather(1, U_idx.unsqueeze(1)).squeeze(1)
            unlabeled_adjust_score = max_unlabeled_score_r

        max_unlabeled_logits_idx = unlabeled_logits.argmax(dim=1)
        return unlabeled_adjust_score, max_unlabeled_logits_idx


class Naive():
    def __call__(self, unlabeled_logits, labeled_logits, labeled_labels, score_function):
        unlabeled_all_score = score_function(unlabeled_logits)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)
        
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx]
        
        return max_unlabeled_score, _
    
class RandomSample():
    def __call__(self, unlabeled_logits, labeled_logits, labeled_labels, score_function):
        labeled_true_score = score_function(labeled_logits, labeled_labels)
        labeled_all_score = score_function(labeled_logits)
        unlabeled_all_score = score_function(unlabeled_logits)
        
        _, max_label_logits_idx = torch.kthvalue(labeled_logits, labeled_all_score.size(1), dim=1)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)
        
        max_labeled_score = labeled_all_score[torch.arange(labeled_all_score.size(0)), max_label_logits_idx]
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx]
        
        score_diff = labeled_true_score - max_labeled_score
        random_indices = torch.randint(0, len(score_diff), (len(max_unlabeled_score),))
        random_score_diff = score_diff[random_indices]
        unlabeled_adjust_score = max_unlabeled_score + random_score_diff
        
        return unlabeled_adjust_score, max_unlabeled_logits_idx
    
class Debias():
    def __call__(self, unlabeled_logits, labeled_logits, labeled_labels, score_function):
        labeled_all_score = score_function(labeled_logits)
        unlabeled_all_score = score_function(unlabeled_logits)
        
        _, max_labeled_logits_idx = torch.kthvalue(labeled_logits, labeled_all_score.size(1), dim=1)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)
        
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx]
        max_labeled_score = labeled_all_score[torch.arange(labeled_all_score.size(0)), max_labeled_logits_idx]
        true_labeled_score = score_function(logits=labeled_logits, label=labeled_labels)
        bias = (true_labeled_score-max_labeled_score).mean()
        
        return max_unlabeled_score + bias, _
    
class UNS_imputeacc(UNS):
    "unlabeled nonconformity score with fix impute acc"
    def __call__(self, unlabeled_logits, unlabeled_labels, labeled_logits, labeled_labels, score_function, impute_acc):
        score_function.randomized = False
        labeled_true_score = score_function(logits=labeled_logits, label=labeled_labels)
        labeled_all_score = score_function(logits=labeled_logits)
        unlabeled_all_score = score_function(logits=unlabeled_logits)
        
        num_classes = labeled_logits.shape[0]
        impute_labeled_idx = generate_imputed_labels(labeled_labels, impute_acc, num_classes, labeled_logits.device)
        impute_unlabeled_idx = generate_imputed_labels(unlabeled_labels, impute_acc, num_classes, unlabeled_logits.device)
        
        max_labeled_score = labeled_all_score[torch.arange(labeled_all_score.size(0)), impute_labeled_idx]
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), impute_unlabeled_idx]
        
        score_diff = labeled_true_score - max_labeled_score
        closest_indices, closest_score_diff = self.cal_closest_indices(max_labeled_score, max_unlabeled_score, score_diff)

        if (self.randomized == False) or (score_function.__class__.__name__ == 'THR'):
            unlabeled_adjust_score = max_unlabeled_score + closest_score_diff
        else:
            score_function.randomized = True
            U_all = torch.rand(unlabeled_logits.shape, device=labeled_logits.device) #N*K
            U_single = U_all[torch.arange(U_all.size(0)), impute_unlabeled_idx] #N*1
            
            unlabeled_all_score_r = score_function(logits=unlabeled_logits, label=None, U=U_all) #N*K
            labeled_logits_all = labeled_logits[closest_indices] #N*K
            labeled_labels_all = labeled_labels[closest_indices] #N*1
            labeled_true_score_all = score_function(logits=labeled_logits_all, label=labeled_labels_all, U=U_single) #N*1
            labeled_all_score_all = score_function(logits=labeled_logits_all, label=None, U=U_all) #N*K
            max_label_logits_idx_all = impute_labeled_idx[closest_indices]
            max_labeled_score_all = labeled_all_score_all[torch.arange(labeled_all_score_all.size(0)), max_label_logits_idx_all]
            max_unlabeled_score_r = unlabeled_all_score_r[torch.arange(unlabeled_all_score_r.size(0)), impute_unlabeled_idx]
            score_diff_r = labeled_true_score_all - max_labeled_score_all
            closest_score_diff_r = score_diff_r
            
            unlabeled_adjust_score = max_unlabeled_score_r + closest_score_diff_r
        
        return unlabeled_adjust_score, impute_unlabeled_idx
    

def generate_imputed_labels(true_labels, impute_acc, num_classes, device):
    total_samples = true_labels.size(0)
    num_correct = int(round(impute_acc * total_samples))
    correct_indices = torch.randperm(total_samples)[:num_correct]
    imputed_labels = true_labels.clone()
    imputed_labels[correct_indices] = true_labels[correct_indices]
    
    mask = torch.ones(total_samples, dtype=torch.bool, device=device)
    mask[correct_indices] = False
    incorrect_indices = torch.nonzero(mask, as_tuple=True)[0]
    
    random_labels = torch.randint(0, num_classes, (incorrect_indices.size(0),), device=device)
    random_labels = random_labels + (random_labels == true_labels[incorrect_indices]).long()
    
    imputed_labels[incorrect_indices] = random_labels
    
    return imputed_labels


class UNSScoresNN(UNS):
    def __init__(self, randomized=False, metric='cosine'):
        super().__init__(randomized)
        self.metric = metric  # 'euclidean' or 'cosine'
        
    def __call__(self, unlabeled_logits, labeled_logits, labeled_labels, score_function):
        score_function.randomized = False
        labeled_true_score = score_function(logits=labeled_logits, label=labeled_labels)
        labeled_all_score = score_function(logits=labeled_logits)
        unlabeled_all_score = score_function(logits=unlabeled_logits)
        
        _, max_label_logits_idx = torch.kthvalue(labeled_logits, labeled_all_score.size(1), dim=1)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)
        
        max_labeled_score = labeled_all_score[torch.arange(labeled_all_score.size(0)), max_label_logits_idx]
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx]
        
        score_diff = labeled_true_score - max_labeled_score
        closest_indices, closest_score_diff = self.cal_closest_indices(labeled_all_score, unlabeled_all_score, score_diff)

        if (self.randomized == False) or (score_function.__class__.__name__ == 'THR'):
            unlabeled_adjust_score = max_unlabeled_score + closest_score_diff
        else:
            score_function.randomized = True
            U_all = torch.rand(unlabeled_logits.shape, device=labeled_logits.device) #N*K
            U_single = U_all[torch.arange(U_all.size(0)), max_unlabeled_logits_idx] #N*1
            
            unlabeled_all_score_r = score_function(logits=unlabeled_logits, label=None, U=U_all) #N*K
            labeled_logits_all = labeled_logits[closest_indices] #N*K
            labeled_labels_all = labeled_labels[closest_indices] #N*1
            labeled_true_score_all = score_function(logits=labeled_logits_all, label=labeled_labels_all, U=U_single) #N*1
            labeled_all_score_all = score_function(logits=labeled_logits_all, label=None, U=U_all) #N*K
            max_label_logits_idx_all = max_label_logits_idx[closest_indices]
            max_labeled_score_all = labeled_all_score_all[torch.arange(labeled_all_score_all.size(0)), max_label_logits_idx_all]
            max_unlabeled_score_r = unlabeled_all_score_r[torch.arange(unlabeled_all_score_r.size(0)), max_unlabeled_logits_idx]
            score_diff_r = labeled_true_score_all - max_labeled_score_all
            closest_score_diff_r = score_diff_r
            
            unlabeled_adjust_score = max_unlabeled_score_r + closest_score_diff_r
        
        return unlabeled_adjust_score, max_unlabeled_logits_idx
    
    def cal_closest_indices(self, labeled_all_score, unlabeled_all_score, score_diff):
        """
        计算 unlabeled_all_score 每一行与 labeled_all_score 所有行的距离，
        找到最近的 labeled score 的索引，并用其对应的 score_diff
        """
        # labeled_all_score: (m, k)
        # unlabeled_all_score: (n, k)

        if self.metric == 'euclidean':
            # 使用欧氏距离：||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            a = unlabeled_all_score  # (n, k)
            b = labeled_all_score    # (m, k)
            a_sq = (a ** 2).sum(dim=1).unsqueeze(1)  # (n, 1)
            b_sq = (b ** 2).sum(dim=1).unsqueeze(0)  # (1, m)
            dist = a_sq + b_sq - 2 * a @ b.T         # (n, m)
        elif self.metric == 'cosine':
            a = F.normalize(unlabeled_all_score, dim=1)
            b = F.normalize(labeled_all_score, dim=1)
            dist = 1 - a @ b.T  # 余弦相似度转为距离
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        closest_indices = dist.argmin(dim=1)  # (n,)
        closest_score_diff = score_diff[closest_indices]  # (n,)

        return closest_indices, closest_score_diff
    
class UNSLogitsNN(UNSScoresNN):
    def __call__(self, unlabeled_logits, labeled_logits, labeled_labels, score_function):
        score_function.randomized = False
        labeled_true_score = score_function(logits=labeled_logits, label=labeled_labels)
        labeled_all_score = score_function(logits=labeled_logits)
        unlabeled_all_score = score_function(logits=unlabeled_logits)

        _, max_label_logits_idx = torch.kthvalue(labeled_logits, labeled_all_score.size(1), dim=1)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)

        max_labeled_score = labeled_all_score[torch.arange(labeled_all_score.size(0)), max_label_logits_idx]
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx]

        score_diff = labeled_true_score - max_labeled_score
        # 传入 logits 用于计算最近邻
        closest_indices, closest_score_diff = self.cal_closest_indices(
            labeled_logits, unlabeled_logits, score_diff
        )

        if (self.randomized == False) or (score_function.__class__.__name__ == 'THR'):
            unlabeled_adjust_score = max_unlabeled_score + closest_score_diff
        else:
            score_function.randomized = True
            U_all = torch.rand(unlabeled_logits.shape, device=labeled_logits.device)
            U_single = U_all[torch.arange(U_all.size(0)), max_unlabeled_logits_idx]

            unlabeled_all_score_r = score_function(logits=unlabeled_logits, label=None, U=U_all)
            labeled_logits_all = labeled_logits[closest_indices]
            labeled_labels_all = labeled_labels[closest_indices]
            labeled_true_score_all = score_function(logits=labeled_logits_all, label=labeled_labels_all, U=U_single)
            labeled_all_score_all = score_function(logits=labeled_logits_all, label=None, U=U_all)
            max_label_logits_idx_all = max_label_logits_idx[closest_indices]
            max_labeled_score_all = labeled_all_score_all[torch.arange(labeled_all_score_all.size(0)), max_label_logits_idx_all]
            max_unlabeled_score_r = unlabeled_all_score_r[torch.arange(unlabeled_all_score_r.size(0)), max_unlabeled_logits_idx]
            score_diff_r = labeled_true_score_all - max_labeled_score_all
            closest_score_diff_r = score_diff_r

            unlabeled_adjust_score = max_unlabeled_score_r + closest_score_diff_r

        return unlabeled_adjust_score, max_unlabeled_logits_idx

class UNSFeatureNN(UNSScoresNN):
    def __call__(self, unlabeled_logits, labeled_logits, labeled_labels,
                 labeled_features, unlabeled_features, score_function):

        score_function.randomized = False
        labeled_true_score = score_function(logits=labeled_logits, label=labeled_labels)
        labeled_all_score = score_function(logits=labeled_logits)
        unlabeled_all_score = score_function(logits=unlabeled_logits)

        _, max_label_logits_idx = torch.kthvalue(labeled_logits, labeled_all_score.size(1), dim=1)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)

        max_labeled_score = labeled_all_score[torch.arange(labeled_all_score.size(0)), max_label_logits_idx]
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx]

        score_diff = labeled_true_score - max_labeled_score
        closest_indices, closest_score_diff = self.cal_closest_indices(
            labeled_features, unlabeled_features, score_diff
        )

        if (self.randomized == False) or (score_function.__class__.__name__ == 'THR'):
            unlabeled_adjust_score = max_unlabeled_score + closest_score_diff
        else:
            score_function.randomized = True
            U_all = torch.rand(unlabeled_logits.shape, device=labeled_logits.device)
            U_single = U_all[torch.arange(U_all.size(0)), max_unlabeled_logits_idx]

            unlabeled_all_score_r = score_function(logits=unlabeled_logits, label=None, U=U_all)
            labeled_logits_all = labeled_logits[closest_indices]
            labeled_labels_all = labeled_labels[closest_indices]
            labeled_true_score_all = score_function(logits=labeled_logits_all, label=labeled_labels_all, U=U_single)
            labeled_all_score_all = score_function(logits=labeled_logits_all, label=None, U=U_all)
            max_label_logits_idx_all = max_label_logits_idx[closest_indices]
            max_labeled_score_all = labeled_all_score_all[torch.arange(labeled_all_score_all.size(0)), max_label_logits_idx_all]
            max_unlabeled_score_r = unlabeled_all_score_r[torch.arange(unlabeled_all_score_r.size(0)), max_unlabeled_logits_idx]
            score_diff_r = labeled_true_score_all - max_labeled_score_all
            closest_score_diff_r = score_diff_r

            unlabeled_adjust_score = max_unlabeled_score_r + closest_score_diff_r

        return unlabeled_adjust_score, max_unlabeled_logits_idx


class UNS_n_num():
    "unlabeled nonconformity score"
    def __init__(self, randomized=False, n_neighbors=1):
        self.randomized = randomized
        self.n_neighbors = n_neighbors  # number of nearest neighbors to consider
        
    def __call__(self, unlabeled_logits, labeled_logits, labeled_labels, score_function):
        score_function.randomized = False
        labeled_true_score = score_function(logits=labeled_logits, label=labeled_labels)
        labeled_all_score = score_function(logits=labeled_logits)
        unlabeled_all_score = score_function(logits=unlabeled_logits)
        
        _, max_label_logits_idx = torch.kthvalue(labeled_logits, labeled_all_score.size(1), dim=1)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)
        
        max_labeled_score = labeled_all_score[torch.arange(labeled_all_score.size(0)), max_label_logits_idx]
        max_unlabeled_score = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx]
        
        score_diff = labeled_true_score - max_labeled_score
        closest_indices, closest_score_diff = self.cal_closest_indices(max_labeled_score, max_unlabeled_score, score_diff)

        # Average the bias of the top n closest neighbors
        bias_avg = closest_score_diff[:self.n_neighbors].mean(dim=0)

        if (self.randomized == False) or (score_function.__class__.__name__ == 'THR'):
            unlabeled_adjust_score = max_unlabeled_score + bias_avg
        else:
            score_function.randomized = True
            U_all = torch.rand(unlabeled_logits.shape, device=labeled_logits.device) #N*K
            U_single = U_all[torch.arange(U_all.size(0)), max_unlabeled_logits_idx] #N*1
            
            unlabeled_all_score_r = score_function(logits=unlabeled_logits, label=None, U=U_all) #N*K
            labeled_logits_all = labeled_logits[closest_indices] #N*K
            labeled_labels_all = labeled_labels[closest_indices] #N*1
            labeled_true_score_all = score_function(logits=labeled_logits_all, label=labeled_labels_all, U=U_single) #N*1
            labeled_all_score_all = score_function(logits=labeled_logits_all, label=None, U=U_all) #N*K
            max_label_logits_idx_all = max_label_logits_idx[closest_indices]
            max_labeled_score_all = labeled_all_score_all[torch.arange(labeled_all_score_all.size(0)), max_label_logits_idx_all]
            max_unlabeled_score_r = unlabeled_all_score_r[torch.arange(unlabeled_all_score_r.size(0)), max_unlabeled_logits_idx]
            score_diff_r = labeled_true_score_all - max_labeled_score_all
            closest_score_diff_r = score_diff_r
            
            unlabeled_adjust_score = max_unlabeled_score_r + closest_score_diff_r
        
        return unlabeled_adjust_score, max_unlabeled_logits_idx
    
    def cal_closest_indices(self, max_labeled_score, max_unlabeled_score, score_diff):
        # Get indices of the top n closest neighbors
        # breakpoint()
        closest_indices = torch.abs(max_labeled_score[:, None] - max_unlabeled_score[None, :]).topk(self.n_neighbors, dim=0, largest=False).indices
        closest_score_diff = score_diff[closest_indices]
        # breakpoint()
            
        return closest_indices, closest_score_diff
