import numpy as np
np.random.seed(123)
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
print(X_train.shape)
from matplotlib import pyplot as plt

import matplotlib
matplotlib.use('Qt4Agg', warn=False)

import pdb;pdb.set_trace()
plt.imshow(X_train[0])
plt.show()
