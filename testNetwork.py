import os
import tensorflow as tf
import matplotlib.pyplot as plt 
import cv2 as cv 
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
current_dir = os.getcwd()

# Import mnist data stored in the following path: current directory -> mnist.npz
from tensorflow.keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data(path=current_dir+'/mnist.npz')
X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation= tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation= tf.nn.softmax))

model.compile(optimizer='Adam', loss="sparse_categorical_crossentropy", metrics=['accuracy'], loss_weights=None,
              sample_weight_mode=None, weighted_metrics=None)
model.fit(X_train, Y_train, epochs=3)
loss,accuracy = model.evaluate(X_test,Y_test)
print(loss)
print(accuracy)

for x in range(0,10):
    img = cv.imread(f'E:\code\AI\Xiangpi-AI-with-board-recognition-via-Rasp-Camera-Module\\test_data\\{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"I guess the number is: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()