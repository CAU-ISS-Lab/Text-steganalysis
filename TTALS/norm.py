from copy import deepcopy

import torch
import torch.nn as nn


class Norm(nn.Module):


    def __init__(self, model, eps=1e-5, momentum=0.1,
                 reset_stats=False, no_stats=False):
        super().__init__()
        self.model = model
        self.model = configure_model(self.model, eps, momentum, reset_stats, no_stats)
        self.model_state = deepcopy(self.model.state_dict())

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


def collect_stats(model):
    """Collect the normalization stats from batch norms.

    Walk the model's modules and collect all batch normalization stats.
    Return the stats and their names.
    """
    stats = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            state = m.state_dict()
            if m.affine:
                del state['weight'], state['bias']
            for ns, s in state.items():
                stats.append(s)
                names.append(f"{nm}.{ns}")
    return stats, names


def configure_model(model, eps, momentum, reset_stats, no_stats):
    """Configure model for adaptation by test-time normalization, specifically for LayerNorm."""
    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            m.eps = eps

    return model
