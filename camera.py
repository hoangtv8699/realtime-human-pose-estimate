import cv2
import numpy as np


def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


c = [
    [108, 125, 150],
    [150, 135, 175],
    [122, 148, 250]
]

o = linear_assignment(c)

print(o)