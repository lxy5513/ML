import keras
import numpy as np
from keras.datasets import mnist
import matplotlib
#实现Matplotlib绘图并保存图像但不显示图形的方法 ssh 服务器
matplotlib.use('agg')
import matplotlib.pyplot as plt
from keras.utils import multi_gpu_model

# Describe the number of classes:
num_class = 10

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dense, Activation, Lambda, Reshape, Flatten
from keras.models import Model
from keras import backend as K

# Custom classifier function:
def classifier_func(x):
    # x.shape=(?,10) K.argmax(x, axis=1)---return:(?,)
    # one_hot 输入为n维的整数向量 输出为(n+1)维的独热码   one_hot 将一个值化为一个概率分布的向量
    return x+x*K.one_hot(K.argmax(x, axis=1), num_classes=num_class)
    # return shape=(?,10)


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


# Deep Learning Model:
inputs = Input(shape=(28, 28, 1))
#Encoder:
conv_1 = Conv2D(32, (3,3), strides=(1, 1))(inputs)
act_1 = Activation('relu')(conv_1)
maxpool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_1)

conv_2 = Conv2D(64, (3,3), strides=(1,1), padding='same')(maxpool_1)
act_2 = Activation('relu')(conv_2)
maxpool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_2)
# Output Shape: 6x6x64

# 把多维数据一维化 常用于从卷积层到全连接层的过渡 out_put.shape=(6,256)
flat_1 = Flatten()(maxpool_2)
# Dense 全连接层
fc_1 = Dense(256)(flat_1)
act_3 = Activation('relu')(fc_1)

fc_2 = Dense(128)(act_3)
act_4 = Activation('relu')(fc_2)

# fc_3 return shape=(?,10)
fc_3 = Dense(num_class)(act_4)

# Lambda 用以对上一层输出施以任何Tensorflow表达式
# function 要实现的函数，该函数仅接受一个变量 即上一层的输出
# output_shape 返回应该返回的值的shape
act_class = Lambda(classifier_func, output_shape=(num_class,))(fc_3)
# Output Shape: (?,10)

# -------------Decoder:
# 256: 隐藏层的节点数 即输出层的shape[-1]
fc_4 = Dense(256)(act_class)
act_5 = Activation('relu')(fc_4)
# shape=(?,256)

fc_5 = Dense(2304)(act_5)
act_6 = Activation('relu')(fc_5)
# shape=(?, 2304)

# Reshape 将输入的shape转为特定的shape
reshape_1 = Reshape((6, 6, 64))(act_6)

upsample_1 = UpSampling2D((2, 2))(reshape_1)
# shape = (?, 12, 12, 64)

deconv_1 = Conv2DTranspose(64, (3, 3), strides=(1, 1))(upsample_1)
# shape=(?, ?, ?, 64)

act_7 = Activation('relu')(deconv_1)

upsample_2 = UpSampling2D((2, 2))(act_7)
# shape=(?, 28, 28, 64)

# 该层是转置的卷积操作（反卷积） 对一个普通的卷积结果做反方向的变换
deconv_2 = Conv2DTranspose(32, (3, 3), strides=(1, 1))(upsample_2)
act_8 = Activation('relu')(deconv_2)
# shape=(?, ?, ?, 32)

conv_3 = Conv2D(1, (3, 3), strides=(1, 1))(act_8)
act_9 = Activation('sigmoid')(conv_3)
# Output Shape: 28x28x1

autoencoder = Model(inputs, act_9)


try:
    sa
    autoencoder = multi_gpu_model(autoencoder, gpus=2)
    print("Training using multiple GPUs..")
except:
    print("Training using single GPU or CPU..")

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()



# define the checkpoint
from keras.callbacks import ModelCheckpoint
filepath="model-{epoch:02d}-{loss:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]


# Training Model:
epochs = 5
batch_size = 256
# validation_data be used to avoiding overfit enhance rebust
autoencoder.fit(X, X, batch_size=batch_size, epochs=epochs, validation_data=(X_test, X_test), shuffle=True)

decoded_imgs = autoencoder.predict(X_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)




# ------------ save the template model rather than the gpu_mode ----------------
# serialize model to JSON
model_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")



# -------------- load the saved model --------------
from keras.models import model_from_json

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")





num_class = 10

# Split autoencoder:
encoder = Model(inputs, act_class)
encoder.summary()

encode = encoder.predict(X)

class_dict = np.zeros((num_class, num_class))
for i, sample in enumerate(Y):
    class_dict[np.argmax(encode[i], axis=0)][sample] += 1

print(class_dict)

neuron_class = np.zeros((num_class))
for i in range(num_class):
    neuron_class[i] = np.argmax(class_dict[i], axis=0)

print(neuron_class)

encode = encoder.predict(X_test)

predicted = np.argmax(encode, axis=1)
for i, sample in enumerate(predicted):
    predicted[i] = neuron_class[predicted[i]]

comparison = Y_test == predicted
loss = 1 - np.sum(comparison.astype(int))/Y_test.shape[0]








print('Loss:', loss)
print('Examples:')
for i in range(10):
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()
    neuron = np.argmax(encode[i], axis=0)
    print('Class:', Y_test[i], '- Model\'s Output Class:', neuron_class[neuron])

