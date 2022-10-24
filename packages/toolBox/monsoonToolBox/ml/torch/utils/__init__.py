from typing import List
import torch.nn as nn

def mergeModelStateDicts(*models: nn.Module) -> dict:
    """Get averaged state dict from models

    Returns:
        dict: state dict, uset model.load_state_dict(...) to load it.
    """
    def averageWeights(weights: List[float]):
        length = len(weights)
        sum = 0
        for w in weights:
            sum += w
        return w/length

    sds = [model.state_dict() for model in models]
    sd0 = sds[0]

    for key in sd0:
        sd0[key] = averageWeights([sd[key] for sd in sds])
    return sd0