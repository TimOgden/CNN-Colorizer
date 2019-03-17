import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import *
from keras.optimizers import Adam
import numpy as np
import cv2

(_,_), (x_test,_) = cifar10.load_data()

def build_unet(pretrained_weights=None, input_size=(32,32,1)):
	inputs = Input(input_size)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = Dropout(0.5)(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
	merge6 = concatenate([drop4,up6], axis = 3)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = 3)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = 3)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = 3)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv9 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

	model = Model(input = inputs, output = conv10)

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_absolute_error', metrics = ['accuracy'])
	
	#model.summary()

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

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

def build_resnet():
	model = Sequential([
		Conv2D(64, (3,3), padding='same', input_shape=(32,32,1), activation='relu', data_format='channels_last'),
		Conv2D(64, (3,3), padding='same', activation='relu'),
		MaxPooling2D(pool_size=2),

		Conv2D(128, (3,3), padding='same', activation='relu'),
		Conv2D(128, (3,3), padding='same', activation='relu'),
		MaxPooling2D(pool_size=2),

		Conv2D(256, (3,3), padding='same', activation='relu'),
		Conv2D(256, (3,3), padding='same', activation='relu'),
		MaxPooling2D(pool_size=2),

		Conv2D(512, (3,3), padding='same', activation='relu'),
		Conv2D(512, (3,3), padding='same', activation='relu'),
		MaxPooling2D(pool_size=2),

		Conv2D(256, (3,3), padding='same', activation='relu'),
		BatchNormalization(),
		UpSampling2D((2,2)),
		Conv2D(128, (3,3), padding='same', activation='relu'),
		BatchNormalization(),
		UpSampling2D((2,2)),
		Conv2D(64, (3,3), padding='same', activation='relu'),
		BatchNormalization(),
		UpSampling2D((2,2)),
		Conv2D(64, (3,3), padding='same', activation='relu'),
		BatchNormalization(),
		Conv2D(1, (3,3), padding='same', activation='relu'),
		UpSampling2D((2,2))
	])
	return model

model = build_unet(pretrained_weights= '../weights/best_same.h5')
#model = build_model()


img = x_test[10]
plt.subplot(1,4,1)
plt.title('Original Image')
plt.imshow(img)

gray_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
normed_gray = gray_x / 255.

plt.subplot(1,4,2)
plt.title('Grayscale Image')
plt.imshow(gray_x, cmap='gray')
print(gray_x.shape)

normed_gray = np.reshape(normed_gray, (-1, gray_x.shape[0], gray_x.shape[1], 1))
print(normed_gray.shape)

recolored = model.predict(normed_gray)*255.
print(recolored[0][0][0])
print(recolored.shape)
#recolored = np.reshape(recolored, (recolored.shape[1], recolored.shape[2], recolored.shape[3]))
recolored = np.reshape(recolored, (recolored.shape[1], recolored.shape[2]))
print(recolored.shape)
#recolored *= 255.
plt.subplot(1,4,3)
plt.title('CNN Prediction')
plt.imshow(recolored, cmap='gray')


plt.subplot(1,4,4)
plt.title('Prediction - Actual')
plt.imshow(recolored - gray_x)
plt.colorbar()
plt.show()