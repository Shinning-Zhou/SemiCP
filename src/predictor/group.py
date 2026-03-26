import torch

from torchcp.classification.predictor import SplitPredictor


class GroupPredictor(SplitPredictor):
    def __init__(self, score_function, model=None, group_num=20, temperature=1):
        super(GroupPredictor, self).__init__(score_function, model, temperature)
        self.q_hat = None
        self.group_num = group_num

    def calculate_threshold(self, logits, labels, groups, alpha):
        if not (0 < alpha < 1):
            raise ValueError("alpha should be a value in (0, 1).")
        alpha = torch.tensor(alpha, device=self._device)
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels)
        self.q_hat = torch.zeros(self.group_num, device=self._device)
        marginal_q_hat = self._calculate_conformal_value(scores, alpha)
        
        for i in range(self.group_num):
            temp_scores = scores[groups == i]
            self.q_hat[i] = self._calculate_conformal_value(temp_scores, alpha, marginal_q_hat)
            
    def predict_with_logits(self, logits, groups):
        logits = logits.to(self._device)
        scores = self.score_function(logits).to(self._device)
        S = torch.zeros_like(logits, device=self._device, dtype=torch.int32)
        for i in range(self.group_num):
            S[i == groups] = self._generate_prediction_set(scores[i == groups], self.q_hat[i])
        return S
        
        