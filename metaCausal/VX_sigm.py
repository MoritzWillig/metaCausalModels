import math

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap

# sigmoidal used in paper
def sigm(x, u=0.5, s=15, m=1.01):
    p = np.exp(-(x - u) * s)
    #y = 1 / (1+p)
    y = (1 / (1+p) - 0.5) * m + 0.5

    g = (1+s*p) / (1+p)
    grad = y*(g-y) * m

    return y, grad
