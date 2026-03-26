import torch
from torchcp.classification.predictor import ClassWisePredictor

class SemiClasswisePredictor(ClassWisePredictor):
    def __init__(self, score_function, unlabeled_score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)
        self.unlabeled_score_function = unlabeled_score_function
        
    def calculate_threshold(self, labeled_logits, labeled_labels, unlabeled_logits, alpha):
        labeled_logits = labeled_logits.to(self._device)
        labeled_labels = labeled_labels.to(self._device)
        unlabeled_logits = unlabeled_logits.to(self._device)
        num_classes = labeled_logits.shape[1]
        self.q_hat = torch.zeros(num_classes, device=self._device)
        
        labeled_scores = self.score_function(labeled_logits, labeled_labels)
        unlabeled_scores, unlabeled_labels = self.unlabeled_score_function(unlabeled_logits, labeled_logits, labeled_labels, self.score_function)
        scores = torch.cat([labeled_scores, unlabeled_scores])
        labels = torch.cat([labeled_labels, unlabeled_labels])
        marginal_q_hat = self._calculate_conformal_value(scores, alpha)
        
        for label in range(num_classes):
            temp_scores = scores[labels == label]
            self.q_hat[label] = self._calculate_conformal_value(temp_scores, alpha, marginal_q_hat)
        