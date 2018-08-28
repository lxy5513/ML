"""
Train a simple deep NN on the MNIST dataset
Run on CPU cosume 2m30s  each epoch consume 8s
on GPU each cosumn 2s
"""

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import datetime

time_start = datetime.datetime.now()

BATCH_SIZE = 128
EPOCHS = 20
NUM_CLASSES = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# 归一化
x_train /= 255  #a==x_train
x_test /= 255
print(x_train.shape[0], x_test.shape[0])

# Convert class vector to binary class metric
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(NUM_CLASSES, activation='softmax'))

# print the statement of model
model.summary()

# metric used to evaluate the performence of model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    verbose=1, validation_data=(x_test, y_test)
                    )
score = model.evaluate(x_test, y_test, verbose=0)

print('Test Loss: ', score[0], '\tTest accuracy: ', score[1])

time_end = datetime.datetime.now()
time = time_end - time_start
print(time)


'''
keras.summary()
    
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 512)               401920    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               262656    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 669,706
    Trainable params: 669,706
    Non-trainable params: 0
    _________________________________________________________________
    Train on 60000 samples, validate on 10000 samples
'''
