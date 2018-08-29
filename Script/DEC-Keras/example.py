from keras_dec import DeepEmbeddingClustering
from keras.datasets import mnist
import numpy as np
import pdb
import tensorflow as tf


def get_mnist():
    np.random.seed(1234) # set seed for deterministic ordering
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_all = np.concatenate((x_train, x_test), axis = 0)
    Y = np.concatenate((y_train, y_test), axis = 0)
    X = x_all.reshape(-1,x_all.shape[1]*x_all.shape[2])
    # p.shape = (70000,0)  打乱排序
    p = np.random.permutation(X.shape[0])
    X = X[p].astype(np.float32)*0.02
    Y = Y[p]

    # X.shape
    # (70000, 784)
    # Y.shape
    # (70000,)
    return X, Y

for d in ['/device:GPU:0', '/device:GPU:1']:
    with tf.device(d):
        pass



X, Y = get_mnist()

c = DeepEmbeddingClustering(n_clusters=10, input_dim=784)
c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000)

# 提供了数据集 和 标签
c.cluster(X, y=Y)
