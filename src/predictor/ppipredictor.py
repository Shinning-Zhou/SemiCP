import torch
import math
from torchcp.classification.predictor import SplitPredictor
from ppi_py import ppi_quantile_pointestimate

class PPIPredictor(SplitPredictor):
    def __init__(self, score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)
        
    def calculate_threshold(self, labeled_logits, labeled_labels, unlabeled_logits, alpha):
        labeled_logits = labeled_logits.to(self._device)
        labeled_labels = labeled_labels.to(self._device)
        unlabeled_logits = unlabeled_logits.to(self._device)
        
        labeled_true_score = self.score_function(labeled_logits, labeled_labels)
        labeled_all_score = self.score_function(labeled_logits)
        unlabeled_all_score = self.score_function(unlabeled_logits)
        
        _, max_label_logits_idx = torch.kthvalue(labeled_logits, labeled_all_score.size(1), dim=1)
        _, max_unlabeled_logits_idx = torch.kthvalue(unlabeled_logits, unlabeled_all_score.size(1), dim=1)
        
        max_labeled_score_np = labeled_all_score[torch.arange(labeled_all_score.size(0)), max_label_logits_idx].cpu().numpy()
        max_unlabeled_score_np = unlabeled_all_score[torch.arange(unlabeled_all_score.size(0)), max_unlabeled_logits_idx].cpu().numpy()
        labeled_true_score_np = labeled_true_score.cpu().numpy()
        
        quantile_value = 1 - alpha
        
        self.q_hat = ppi_quantile_pointestimate(labeled_true_score_np, max_labeled_score_np, max_unlabeled_score_np, q=quantile_value)