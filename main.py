import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = [i for i in range(-2, 10)]
ys = [2*i - 1 for i in range(-2, 10)]

print(xs)
print(ys)

model.fit(xs, ys, epochs=1000)

print(model.predict([300.0]))