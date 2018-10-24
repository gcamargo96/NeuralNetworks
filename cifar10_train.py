import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

epochs = 10
batch_size = 128
learning_rate = 0.001

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

num_classes = 10
class_names =  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

train_images = train_images.reshape(len(train_images), 3,32,32).transpose([0, 2, 3, 1])
test_images = test_images.reshape(len(test_images), 3,32,32).transpose([0, 2, 3, 1])

model = keras.Sequential([
    keras.layers.Conv2D(100, kernel_size=(2, 2), activation='relu', input_shape = (32, 32, 3) ),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(100, kernel_size=(2, 2), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(200, kernel_size=(2, 2), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    keras.layers.BatchNormalization(),

    keras.layers.Flatten(),

    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate),
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
         )

model.save("model.h5")

test_loss, test_acc = model.evaluate(test_images, test_labels)
 
print("Test accuracy = ", test_acc)