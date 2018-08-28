"""基于多层感知器的softmax多分类"""
import pdb
import keras
from keras.optimizers import SGD #随机梯度下降
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# ###Generate dummy(虚拟） data
import numpy as np


x_train = np.random.random((1000, 20))
# Convert a class vector(integer) to binary class matrix eg:for use with categorical_crossentroy
y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10) #shape=(1000, 10)
x_test = np.random.random((100, 20))
y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)


# Sequential 序贯模型是个多个网络层的线性堆叠 也就是"一条路走到黑"
model = Sequential()
# dense 表示一层全连接层神经网络 32 represent hidden units
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=200, batch_size=128)
# pdb.set_trace()
# score = model.evaluate(x_test, y_test, batch_size=128)
#
"""
补充：
    Momentum
        SDG在其更新的方向完全依赖当前batch计算的速度 十分不稳定
        momentum 模拟的是运动物体的惯性 在更新的时候在一定程度上保留之前更新的方向 同时利用当前batch的梯度微调最终的更新方向
    Nesterov Momentum
        在小球下降的过程中 希望小球提前知道在哪些地方坡面会上升 在上升之前 小球就开始减速
    metrics['accuracy']
        对于分类问题， 我们一般将该列表设置为matrics=['accuracy'] 指标可以是预定义指标的名字， 也可以是用户自定义的函数 指标函数应该
        返回单个张量 或者完成matric_name --> matric_value的映射
"""
