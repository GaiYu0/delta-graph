import torch as th

mse = lambda y, y_bar: th.mean((y - y_bar) ** 2)
rmse = lambda y, y_bar: mse(y, y_bar) ** 0.5
