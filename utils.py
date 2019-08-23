import torch as th
import torch.nn.functional as F

rmse_loss = lambda x, y: th.sqrt(F.mse_loss(x, y))
