import torch
import math
import numpy as np
import random
import torch
import warnings
from torchcp.classification.predictor import SplitPredictor
from .semisplit import SemiPredictor

class InterPredictor(SplitPredictor):
    def _calculate_conformal_value(self, scores, alpha, marginal_q_hat=torch.inf):
        return calculate_conformal_value_linear(scores, alpha, marginal_q_hat)
    

class SemiInterPredictor(SemiPredictor):
    def _calculate_conformal_value(self, scores, alpha, marginal_q_hat=torch.inf):
        return calculate_conformal_value_linear(scores, alpha, marginal_q_hat)

    
    
def calculate_conformal_value_linear(scores, alpha, default_q_hat=torch.inf):
    if default_q_hat == "max":
        default_q_hat = torch.max(scores)
    if alpha >= 1 or alpha <= 0:
        raise ValueError("Significance level 'alpha' must be in [0,1].")
    if len(scores) == 0:
        warnings.warn(
            f"The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as {default_q_hat}.")
        return default_q_hat
    N = scores.shape[0]
    quantile_value = 1 - alpha

    return torch.quantile(scores, quantile_value, dim=0, interpolation='linear').to(scores.device)
