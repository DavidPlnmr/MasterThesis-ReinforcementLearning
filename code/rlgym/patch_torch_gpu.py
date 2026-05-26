import torch

# Patch torch.load pour forcer CPU sur machine sans GPU
_original_torch_load = torch.load
def _cpu_load(*args, **kwargs):
    kwargs.setdefault('map_location', torch.device('cpu'))
    return _original_torch_load(*args, **kwargs)
torch.load = _cpu_load