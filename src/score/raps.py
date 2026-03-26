import torch

from .aps import APS


class RAPS(APS):
    def __init__(self, score_type="softmax", randomized=True, penalty=0, kreg=0):

        super().__init__(score_type=score_type, randomized=randomized)
        if penalty < 0:
            raise ValueError("The parameter 'penalty' must be a nonnegative value.")

        if type(kreg) != int or kreg < 0:
            raise TypeError("The parameter 'kreg' must be a nonnegative integer.")
        self.__penalty = penalty
        self.__kreg = kreg
        
    def __call__(self, logits, label=None, U=None):
        if len(logits.shape) > 2:
            raise ValueError("dimension of logits are at most 2.")

        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        probs = self.transform(logits)
        
        if label is None:
            return self._calculate_all_label(probs, U)
        else:
            return self._calculate_single_label(probs, label, U)

    def _calculate_all_label(self, probs, U):
        indices, ordered, cumsum = self._sort_sum(probs)
        
        if self.randomized == False:
            U = torch.zeros_like(probs)
        elif U != None:
            pass
        else:
            U = torch.rand(probs.shape, device=probs.device)
            
        reg = torch.maximum(self.__penalty * (torch.arange(1, probs.shape[-1] + 1, device=probs.device) - self.__kreg),
                            torch.tensor(0, device=probs.device))
        ordered_scores = cumsum - ordered * U + reg
        _, sorted_indices = torch.sort(indices, descending=False, dim=-1)
        scores = ordered_scores.gather(dim=-1, index=sorted_indices)
        return scores

    def _calculate_single_label(self, probs, label, U):
        indices, ordered, cumsum = self._sort_sum(probs)
        
        if self.randomized == False:
            U = torch.zeros(probs.shape[0], device=probs.device)
        elif U != None:
            pass
        else:
            U = torch.rand(probs.shape[0], device=probs.device)
            
        idx = torch.where(indices == label.view(-1, 1))
        reg = torch.maximum(self.__penalty * (idx[1] + 1 - self.__kreg), torch.tensor(0).to(probs.device))
        scores = cumsum[idx] - U * ordered[idx] + reg
        return scores
