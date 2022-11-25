import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Load MNIST dataset
MNIST = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = MNIST.load_data()

# reshape, normalize and convert to float32
x_train = (x_train / 255.0).reshape([-1, 784]).astype(np.float32)
x_test = (x_test / 255.0).reshape([-1, 784]).astype(np.float32)

# convert labels to one-hot vector
y_train, y_test = tf.one_hot(y_train, 10), tf.one_hot(y_test, 10)

# prepare for training
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(10000).batch(64)

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(train_data)


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


pred = model(x_test)
print(f'Test Accuracy: {accuracy(pred, y_test)}')

# reshape 1 test image
image_index = 4444
test_image = x_test[image_index].reshape(1, 784)

# predict
prediction = model.predict(test_image)
print(tf.round(prediction))

plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
plt.show()
