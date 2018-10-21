import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# %matplotlib notebook

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def transform_to_RGB(img_array):
    return img_array.reshape(len(img_array), 3,32,32).transpose([0, 2, 3, 1])

def normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x-min_val) / (max_val-min_val)
    return x

# Definindo número de épocas
epochs = 10
batch_size = 128
# Definindo tamanho das imagens
channels, img_rows, img_cols = 3, 32, 32
img_shape = (channels, img_rows, img_cols)
input_shape = img_shape

### CIFAR-10 dataset ###

# Carregando imagens
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

learning_rate = 0.001
# Definindo informações das classes
num_classes = 10
class_names =  ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

train_images = normalize(train_images)
test_images = normalize(test_images)

plt.figure(figsize=(3,3))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[np.argmax(train_labels[i])])
plt.show()

model = keras.Sequential([
    keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape = (32, 32, 3) ),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(),

    keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.BatchNormalization(),

    keras.layers.Flatten(),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.BatchNormalization(),

    keras.layers.Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate),
              metrics=['accuracy'])


# checkpoint = keras.callbacks.ModelCheckpoint("model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
model.save("model.h5")

model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks=callbacks_list
         )

test_loss, test_acc = model.evaluate(test_images, test_labels)
 
print('Test accuracy:', test_acc)