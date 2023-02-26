import numpy
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('TkAgg')
plt.interactive(False)

mnist = tf.keras.datasets.fashion_mnist;
(train_images, train_labels), (test_images, test_labels) = mnist.load_data();

index = 1

numpy.set_printoptions(linewidth=320)

train_images = train_images / 255.0
test_images = test_images / 255.0

print(f'LABEL: {test_labels[index]}')
print(f'\nImage Pixel Array:\n {test_images[index]}')

plt.imshow(train_images[index], cmap='Greys')


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

inputs = numpy.array([[1.0, 3.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f'input softmax function: {inputs}')

outputs = tf.keras.activations.softmax(inputs)
print(f'output softmax function: {outputs.numpy()}')

sum = tf.reduce_sum(outputs)
print(f'out sum: {sum}')

prediction = np.argmax(outputs)
print(f'class with highest probability: {prediction}')


model.fit(train_images, train_labels, epochs=10)

print(max(model.predict(numpy.reshape(test_images[index], (1, 28, 28)))[0]))

while True:
    index = int(input('Input index of test element'))
    plt.imshow(test_images[index], cmap='Greys')
    print(model.predict(numpy.reshape(test_images[index], (1, 28, 28)))[0])
    print(max(model.predict(numpy.reshape(test_images[index], (1, 28, 28)))[0]))
    plt.show()