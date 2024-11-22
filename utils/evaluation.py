import torch

def compute_rmse(predictions, targets):
    """
    Calcula el Error Cuadrático Medio (RMSE) entre las valoraciones predichas y las reales.

    Args:
        predictions (torch.Tensor): Valoraciones predichas.
        targets (torch.Tensor): Valoraciones reales.

    Returns:
        rmse (float): Error cuadrático medio.
    """
    mask = targets > 0
    if torch.sum(mask) > 0:
        mse = torch.nn.functional.mse_loss(predictions[mask], targets[mask])
        rmse = torch.sqrt(mse).item()
    else:
        rmse = 0
    return rmse
