import torch
from torchcp.classification.predictor import SplitPredictor

class SemiPredictor(SplitPredictor):
    def __init__(self, score_function, unlabeled_score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)
        self.unlabeled_score_function = unlabeled_score_function
        
    def calculate_threshold(self, labeled_logits, labeled_labels, unlabeled_logits, alpha):
        if labeled_logits is not None:
            labeled_logits = labeled_logits.to(self._device)
            labeled_labels = labeled_labels.to(self._device)
            labeled_scores = self.score_function(labeled_logits, labeled_labels)
            
        unlabeled_logits = unlabeled_logits.to(self._device)
        unlabeled_scores, _ = self.unlabeled_score_function(unlabeled_logits, labeled_logits, labeled_labels, self.score_function)
        
        if labeled_logits is not None:
            scores = torch.cat([labeled_scores, unlabeled_scores])
        else:
            scores = unlabeled_scores
        self.q_hat = self._calculate_conformal_value(scores, alpha)
        
class SemiPredictor_imputeacc(SplitPredictor):
    def __init__(self, score_function, unlabeled_score_function, model=None, temperature=1):
        super().__init__(score_function, model, temperature)
        self.unlabeled_score_function = unlabeled_score_function
        
    def calculate_threshold(self, labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, alpha, impute_acc):
        labeled_logits = labeled_logits.to(self._device)
        labeled_labels = labeled_labels.to(self._device)
        unlabeled_logits = unlabeled_logits.to(self._device)
        labeled_scores = self.score_function(labeled_logits, labeled_labels)
        unlabeled_scores, _ = self.unlabeled_score_function(unlabeled_logits, unlabeled_labels, labeled_logits, labeled_labels, self.score_function, impute_acc)
        scores = torch.cat([labeled_scores, unlabeled_scores])
        self.q_hat = self._calculate_conformal_value(scores, alpha)
        
class SemiPredictor_feature(SemiPredictor):
    def calculate_threshold(self, labeled_logits, labeled_labels, unlabeled_logits, labeled_features, unlabeled_features, alpha):
        labeled_logits = labeled_logits.to(self._device)
        labeled_labels = labeled_labels.to(self._device)
        unlabeled_logits = unlabeled_logits.to(self._device)
        labeled_scores = self.score_function(labeled_logits, labeled_labels)
        unlabeled_scores, _ = self.unlabeled_score_function(unlabeled_logits, labeled_logits, labeled_labels, labeled_features, unlabeled_features, self.score_function)
        scores = torch.cat([labeled_scores, unlabeled_scores])
        self.q_hat = self._calculate_conformal_value(scores, alpha)