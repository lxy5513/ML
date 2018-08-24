import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import os

import numpy as np
np.random.seed(1337)

# 获取当前路径的目录
PATH = os.path.abspath(__file__).split('/')
PATH_DIR = '/'.join(PATH[:-2])

