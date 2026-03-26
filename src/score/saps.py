import torch

from .aps import APS


class SAPS(APS):

    def __init__(self, score_type="softmax", randomized=True, weight=0.2):
        super().__init__(score_type, randomized)
        if weight <= 0:
            raise ValueError("The parameter 'weight' must be a positive value.")
        if not isinstance(randomized, bool):
            raise ValueError("The parameter 'randomized' must be a boolean.")
        self.__weight = weight

    def _calculate_all_label(self, probs, U):
        indices, ordered, cumsum = self._sort_sum(probs)
        ordered[:, 1:] = self.__weight
        cumsum = torch.cumsum(ordered, dim=-1)
        
        if self.randomized == False:
            U = torch.zeros_like(probs)
        elif U != None:
            pass
        else:
            U = torch.rand(probs.shape, device=probs.device)
            
        ordered_scores = cumsum - ordered * U
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
        scores_first_rank = U * cumsum[idx]
        scores_usual = self.__weight * (idx[1] - U) + ordered[:, 0]
        return torch.where(idx[1] == 0, scores_first_rank, scores_usual)
