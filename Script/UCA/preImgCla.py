import keras
import numpy as np
from keras.datasets import mnist
import matplotlib
#实现Matplotlib绘图并保存图像但不显示图形的方法 ssh 服务器
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model


epochs = 3
batch_size = 100
# Getting Dataset:
def get_dataset():
    (X, Y), (X_test, Y_test) = mnist.load_data()
    X = X.astype('float32') / 255.
    X_test = X_test.astype('float32') / 255.
    X = np.reshape(X, (len(X), 28, 28, 1))
    X_test = np.reshape(X_test, (len(X_test), 28, 28, 1))

    # Add noise:  是用来验证训练图片编码模块的吗
    noise_factor = 0.4
    X_train_noisy = X + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    X_test_noisy = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)

    # numpy.clip(a, a_min, a_max, out=None) 限制最小 最大值
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)
    X_test_noisy = np.clip(X_test_noisy, 0., 1.)

    return X, X_test, Y, Y_test, X_train_noisy, X_test_noisy

X, X_test, Y, Y_test, X_train_noisy, X_test_noisy = get_dataset()

import time
time.sleep(0.1)

# About Dataset:
print('Training shape:', X.shape)
print(X.shape[0], 'sample,', X.shape[1], 'x', X.shape[2], 'size grayscale image.\n')
print('Test shape:', X_test.shape)
print(X_test.shape[0], 'sample,', X_test.shape[1], 'x', X_test.shape[2], 'size grayscale image.\n')



print('\n\nExamples:')
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(X_train_noisy[i].reshape(28, 28))
    # import pdb;pdb.set_trace()
    # 显示灰度图
    plt.gray()
    # 设置横坐标不显示
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
# Deep Learning Model:
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.models import Model

# channels_last模式
input_img = Input(shape=(28, 28, 1))

# 二维卷积层 即对图像的空域卷积 该层对二维输出进行滑动窗卷积 当该使用层作为第一层时 应提供input_shape参数
# eg input_shape = (128,128,3) 代表128*128的彩色RGB图像
# filter：16 卷积核的数目（输出的维度）kerbel_size:(3,3) 卷积核的宽度和长度
# strides 默认为(1,1)
# 输入为(sample, rows, cols, channels)4D张量  输出为(sample, new_rows, new_cols, nd_filter)  后面三个变了
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
# 二维池化层 为空域信号施加最大层池化 pool_size=(2,2) 使的图片在两个维度上均变为原来的一半
# 输入(sample, rows, cols, channels)4D张量  输出为(sample, pooled_rows, pooled_cols, channels) 只有中间两个变了
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
# Output Shape: 4x4x8

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
# size=（2，2） 将数据的行和列分别重复size[0] size[1]次  function ????
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
# Output Shape: 28x28x1


# 这个模型是用来把以上的多个神经网络层集合在一起的 function???
autoencoder = Model(input_img, decoded)

try:
    autoencoder = multi_gpu_model(autoencoder, gpus=2)
    print("Training using multiple GPUs..")
except:
    print("Training using single GPU or CPU..")

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

# Checkpoints:
from keras.callbacks import ModelCheckpoint, TensorBoard
checkpoints = []
#checkpoints.append(TensorBoard(log_dir='/Checkpoints/logs'))




'''
# Creates live data:
# For better yield. The duration of the training is extended.

from keras.preprocessing.image import ImageDataGenerator
# 图片生成器 用以生成一个batch的图像数据 支持实时数据提升 width_shift_range 数据提升时图片水平偏移的幅度 horizontal_flip 进行随机水平翻转
generated_data = ImageDataGenerator(featurewise_center=False, samplewise_center=False, featurewise_std_normalization=False,
                                    samplewise_std_normalization=False, zca_whitening=False, rotation_range=0,  width_shift_range=0.1,
                                    height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
generated_data.fit(X_train_noisy)
# flow 接受numpy数组和标签为参数 生成进过数据提升或标准化后的batch数据 并在一个无限循环中不断返回batch数据
# 利用python生成器，逐个生成数据的batch进行训练，生成器与模型将并行执行以提高效率
#n=steps_per_epoch,则当生成器返回n次数据时计一个epoch结束 执行下一个epoch
# 返回一个history对象 回掉函数
autoencoder.fit_generator(generated_data.flow(X_train_noisy, X, batch_size=batch_size), steps_per_epoch=X.shape[0],
                          epochs=epochs, validation_data=(X_test_noisy, X_test), callbacks=checkpoints)
'''





# Training Model:
epochs = 3
batch_size = 100
# 该回掉函数将在每一个epoch后保存模型到filepath  有的回掉函数将以字典logs为参数 将包含当前batch 或 epoch的相关信息 logs将包含训练的正确率和误差
autoencoder.fit(X_train_noisy, X, batch_size=batch_size, epochs=epochs, validation_data=(X_test_noisy, X_test), shuffle=True, callbacks=checkpoints)

# 按Batch获得的输入数据进行相应的输出
decoded_imgs = autoencoder.predict(X_test_noisy)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)



plt.show()
