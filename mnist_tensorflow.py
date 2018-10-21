import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Constantes usadas no programa
epochs = 1
nclasses = 10
nrows, ncols = 28, 28

# Carregando o dataset mnist
mnist = keras.datasets.mnist

# Separando os dados em treinamento e testes
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Adequando os conjuntos ao modelo de imagem do keras
if keras.backend.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, nrows, ncols)
    test_images = test_images.reshape(test_images.shape[0], 1, nrows, ncols)
    input_shape = (1, nrows, ncols)
else:
    train_images = train_images.reshape(train_images.shape[0], nrows, ncols, 1)
    test_images = test_images.reshape(test_images.shape[0], nrows, ncols, 1)
    input_shape = (nrows, ncols, 1)

# Tornando os labels categoricos
train_labels = keras.utils.to_categorical(train_labels, nclasses)
test_labels = keras.utils.to_categorical(test_labels, nclasses)

# Normalizacao dos conjuntos de treinamento e testes
train_images = train_images/255.0
test_images = test_images/255.0

# Abrindo o arquivo de saida que guardará os resultados dos testes
outfile = open('results.txt', 'w')

# Rede com 1 camada convolucional com 24 filtros 3x3
model = keras.Sequential([
    keras.layers.Conv2D(24, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          epochs=epochs,
          verbose=1,
         )

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
s = 'Network 1: ' + str(test_accuracy) + '\n'
outfile.write(s)


# Rede com 1 camada convolucional com 48 filtros 3x3
model = keras.Sequential([
    keras.layers.Conv2D(48, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          epochs=epochs,
          verbose=1,
         )

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
s = 'Network 2: ' + str(test_accuracy) + '\n'
outfile.write(s)


# Rede com 1 camada convolucional com 24 filtros 6x6
model = keras.Sequential([
    keras.layers.Conv2D(24, kernel_size=(6, 6), activation='relu', input_shape=input_shape),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          epochs=epochs,
          verbose=1,
         )

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
s = 'Network 3: ' + str(test_accuracy) + '\n'
outfile.write(s)


# Rede com 2 camadas convolucionais com 24 filtros 3x3 cada
model = keras.Sequential([
    keras.layers.Conv2D(24, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    keras.layers.Conv2D(24, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(nclasses, activation='softmax')
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          epochs=epochs,
          verbose=1,
         )

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
s = 'Network 4: ' + str(test_accuracy) + '\n'
outfile.write(s)


# Rede com 2 camadas convolucionais com 48 filtros 3x3 cada
model = keras.Sequential([
    keras.layers.Conv2D(48, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
    keras.layers.Conv2D(48, kernel_size=(3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(nclasses, activation='softmax')
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          epochs=epochs,
          verbose=1,
         )

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
s = 'Network 5: ' + str(test_accuracy) + '\n'
outfile.write(s)


# Rede com 2 camadas convolucionais com 24 filtros 6x6 cada
model = keras.Sequential([
    keras.layers.Conv2D(24, kernel_size=(6, 6), activation='relu', input_shape=input_shape),
    keras.layers.Conv2D(24, kernel_size=(6, 6), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(nclasses, activation='softmax')
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          epochs=epochs,
          verbose=1,
         )

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
s = 'Network 6: ' + str(test_accuracy) + '\n'
outfile.write(s)


# Rede com 1 camada convolucional com 24 filtros 3x3 e uma camada densa de 100 neurônios
model = keras.Sequential([
    keras.layers.Conv2D(24, kernel_size=(6, 6), activation='relu', input_shape=input_shape),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(nclasses, activation='softmax')
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          epochs=epochs,
          verbose=1,
         )

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
s = 'Network 7: ' + str(test_accuracy) + '\n'
outfile.write(s)

outfile.close()