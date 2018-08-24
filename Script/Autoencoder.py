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

def Func(epoch=30):
    # y_test是标签集
    (x_train, _), (x_test, y_test) = mnist.load_data()

    print("The shape of 测试集 is\n", x_test.shape, '\n', y_test.shape)

    # data pre_processing
    x_train = x_train.astype('float32')/255. - 0.5   # minimax_normolized
    x_test = x_test.astype('float32')/255. - 0.5   # minimax_normolized

    x_train = x_train.reshape((x_train.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    print(x_train.shape)
    print(x_test.shape)


    # in order to plot in a 2D figure
    encoding_dim = 2

    # this is our input placehoder
    input_img = Input(shape=(784,))

    # encoder layers
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim)(encoded)

    # decoder layers
    decoded = Dense(10, activation='relu')(encoder_output)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='tanh')(decoded)

    # construct the autoencoder model
    autoencoder = Model(input=input_img, output=decoded)

    # construct the encoder for plt plotting
    encoder = Model(input=input_img, output=encoder_output)

    # compile autoencoder
    autoencoder.compile(optimizer='adam', loss='mse')

    # training
    autoencoder.fit(x_train, x_train, nb_epoch=epoch, batch_size=256, shuffle=True)


    # plotting
    encoded_imgs = encoder.predict(x_test)
    plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test, marker='.')
    plt.colorbar()

    P_DIR = PATH_DIR + '/Image'
    name = 'auto{}epoch'.format(epoch)
    if not os.path.exists(P_DIR):
        os.mkdir(P_DIR)

    plt.savefig('{}/{}.png'.format(P_DIR, name))
    # create image direction

    plt.show()

if __name__ == '__main__':
    Func(10)
