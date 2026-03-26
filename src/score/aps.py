import torch

from torchcp.classification.score.thr import THR


class APS(THR):
    def __init__(self, score_type="softmax", randomized=True):
        super().__init__(score_type)
        self.randomized = randomized
        
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
        if probs.dim() == 1 or probs.dim() > 2:
            raise ValueError("Input probabilities must be 2D.")
        indices, ordered, cumsum = self._sort_sum(probs)
        
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

    def _sort_sum(self, probs):
        ordered, indices = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(ordered, dim=-1)
        return indices, ordered, cumsum

    def _calculate_single_label(self, probs, label, U):
        indices, ordered, cumsum = self._sort_sum(probs)
        
        if self.randomized == False:
            U = torch.zeros(probs.shape[0], device=probs.device)
        elif U != None:
            pass
        else:
            U = torch.rand(probs.shape[0], device=probs.device)
            
        idx = torch.where(indices == label.view(-1, 1))
        scores = cumsum[idx] - U * ordered[idx]
        return scores
