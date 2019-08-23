import torch as th
import torch.nn.functional as F

rmse_loss = lambda x: th.sqrt(F.mse_loss(x))
