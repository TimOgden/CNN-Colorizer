import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam, Nadam
import numpy as np
import cv2
from colorizor import build_unet
import time

(_,_), (x_test,_) = cifar10.load_data()

model = build_unet(pretrained_weights= '../weights/best.h5')
#model = build_model()
start_index = 30
n_rows = 5
c = 1
for r in range(n_rows):
	img = x_test[r+start_index]
	gray_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	normed_gray = gray_x / 255.

	plt.subplot(n_rows,4,c)
	plt.title('Grayscale Image')
	plt.imshow(gray_x, cmap='gray')
	print(gray_x.shape)
	c+=1

	plt.subplot(n_rows,4,c)
	plt.title('Original Image')
	plt.imshow(img)
	c+=1

	normed_gray = np.reshape(normed_gray, (-1, gray_x.shape[0], gray_x.shape[1], 1))

	start = time.time()
	recolored = model.predict(normed_gray)
	end = time.time() - start
	print('Took {} seconds'.format(end))
	recolored = np.reshape(recolored, (recolored.shape[1], recolored.shape[2], recolored.shape[3]))
	#recolored = np.reshape(recolored, (recolored.shape[1], recolored.shape[2]))
	#recolored *= 255.
	plt.subplot(n_rows,4,c)
	plt.title('CNN Prediction')
	plt.imshow(recolored, cmap='gray')
	c+=1

	plt.subplot(n_rows,4,c)
	plt.title('Prediction - Actual')
	plt.imshow(np.absolute(recolored - img))
	c+=1
	#plt.colorbar()
plt.show()