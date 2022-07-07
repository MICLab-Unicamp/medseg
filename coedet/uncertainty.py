import torch
from torch.nn import Module
from torch import Tensor
from typing import Tuple
from tqdm import tqdm


def get_epistemic_uncertainty(model: Module, x: Tensor, n: int = 10) -> Tuple[Tensor, Tensor]:
    '''
    Estimates epistemic uncertainty with n monte carlo predictions of model on x.

    Returns:
        standard deviation uncertainty, mean prediction
    '''
    model = model.train()
    with torch.no_grad():
        uncertain_preds = [model(x).detach().cpu() for _ in tqdm(range(n), leave=False)]
    model = model.eval()

    uncertain_preds_tensor = torch.stack(uncertain_preds)
    epistemic_uncertainty = uncertain_preds_tensor.std(dim=0)
    mean_prediction = uncertain_preds_tensor.mean(dim=0)
    
    return epistemic_uncertainty, mean_prediction
