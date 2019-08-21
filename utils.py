import torch as th

rmse = lambda y, y_bar: (th.mean((y - y_bar) ** 2)) ** 0.5
