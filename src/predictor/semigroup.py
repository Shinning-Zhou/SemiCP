import torch

from .group import GroupPredictor


class SemiGroupPredictor(GroupPredictor):
    def __init__(self, score_function, unlabeled_score_function, model=None, group_num=20, temperature=1):
        super().__init__(score_function, model, group_num, temperature)
        self.unlabeled_score_function = unlabeled_score_function
        
    def calculate_threshold(self, labeled_logits, labeled_labels, labeled_groups, unlabeled_logits, unlabeled_groups, alpha):
        labeled_logits = labeled_logits.to(self._device)
        labeled_labels = labeled_labels.to(self._device)
        labeled_groups = labeled_groups.to(self._device)
        unlabeled_logits = unlabeled_logits.to(self._device)
        unlabeled_groups = unlabeled_groups.to(self._device)
        alpha = torch.tensor(alpha, device=self._device)
        
        labeled_scores = self.score_function(labeled_logits, labeled_labels)
        unlabeled_scores, _ = self.unlabeled_score_function(unlabeled_logits, labeled_logits, labeled_labels, self.score_function)
        scores = torch.cat([labeled_scores, unlabeled_scores])
        groups = torch.cat([labeled_groups, unlabeled_groups])
        
        self.q_hat = torch.zeros(self.group_num, device=self._device)
        marginal_q_hat = self._calculate_conformal_value(scores, alpha)
        
        for i in range(self.group_num):
            temp_scores = scores[groups == i]
            self.q_hat[i] = self._calculate_conformal_value(temp_scores, alpha, marginal_q_hat)
        