import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, LeakyReLU, Dropout, MaxPooling2D, UpSampling2D
import numpy as np
import cv2

(_,_), (x_test,_) = cifar10.load_data()

def build_model():
	dropout = .2
	model = Sequential([
			Conv2D(32, (3,3), padding='same', input_shape=(32,32,1)),
			LeakyReLU(),
			Dropout(dropout),
			MaxPooling2D((2,2), padding='same'),

			Conv2D(64, (3,3), padding='same'),
			LeakyReLU(),
			Dropout(dropout),

			UpSampling2D(),
			Conv2D(3, (3,3), padding='same'),
		])
	return model

model = build_model()
model.load_weights('../weights/best.h5')
img = x_test[10]
normed_img = img / 255.
plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(img)

gray_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.subplot(1,3,2)
plt.title('Grayscale Image')
plt.imshow(gray_x, cmap='gray')
print(gray_x.shape)

gray_x = np.reshape(gray_x, (-1, gray_x.shape[0], gray_x.shape[1], 1))
print(gray_x.shape)

recolored = model.predict(gray_x)
print(recolored[0][0][0])
print(recolored.shape)
recolored = np.reshape(recolored, (recolored.shape[1], recolored.shape[2], recolored.shape[3]))
#recolored *= 255.
plt.subplot(1,3,3)
plt.title('CNN Prediction')
plt.imshow(recolored)
plt.show()