"""
Train a simple conv net on the MNIST dataset
on GPU each epoch consumes 185s
"""

import keras
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

BATCH_SIZE = 128
# result from 0 to 9
NUM_CLASSES = 10
EPOCHS = 12

# input image dimentions
img_rows, img_cols = 28, 28

# the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(K.image_data_format())

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train /= 255
x_test /= 255
print("x_train shape", x_train.shape)
print('sample_num: \n', x_train.shape[0], x_test.shape[0])

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(63, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy']
              )

model.fit(
    x_train, y_train,
    epochs=EPOCHS, verbose=1,
    validation_data=(x_test, y_test)
)

score = model.evaluate(x_test, y_test, verbose=0)

print('Test Loss: ', score[0])
print('Test Accuracy: ', score[1])
