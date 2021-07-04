import torch
import time
import numpy as np
x = torch.rand(2, 3)
y = torch.rand(3, 1)

print(torch.mm(x, y).squeeze(1))