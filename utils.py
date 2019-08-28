import torch as th
import torch.nn.functional as F

rmse_loss = lambda x, y: th.sqrt(F.mse_loss(x, y))

def detach(x):
    if isinstance(x, th.Tensor):
        return x.detach()
    elif isinstance(x, (list, tuple)):
        return [t.detach() for t in x]
    else:
        return x
