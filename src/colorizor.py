from keras.datasets import cifar10
from keras.layers import *
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import numpy as np
import keras
from math import ceil
import time
import cv2
(x_train, _), (x_test, _) = cifar10.load_data()
print(x_train.shape, x_test.shape)
#x_train = np.reshape(x_train, (-1, x_train.shape[0], x_train.shape[1], x_train.shape[2]))
#x_test = np.reshape(x_test, (-1, x_test.shape[0], x_test.shape[1], x_train.shape[2]))
print(x_train.shape, x_test.shape)
batch_size = 64
#categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_rand_imgs():
	rand = np.random.randint(0,100)
	for i in range(10):
		plt.imshow(x_train[i+int(rand)])
		plt.show()


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
	conv10 = Conv2D(3, 1, activation = 'relu')(conv9)

	model = Model(input = inputs, output = conv10)

	model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_absolute_error', metrics = ['accuracy'])
	
	#model.summary()

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

def build_model():
	dropout = .2
	model = keras.Sequential([
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
	model = keras.Sequential([
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
		Conv2D(3, (3,3), padding='same', activation='relu'),
		UpSampling2D((2,2))
	])
	return model

def build_perceptron():
	model = Sequential([
		Flatten(input_shape=(28,28)),
		Dense(28*28, activation='relu'),
		Dense(128, activation='relu'),
		Dense(64, activation='relu'),
		Dense(32, activation='relu'),
		Dense(10, activation='softmax')
	])
	return model

def generator(dataset):
	while True:
		for x in dataset:
			gray_x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
			gray_x = np.reshape(gray_x, (-1, gray_x.shape[0], gray_x.shape[1]))
			gray_x = np.expand_dims(gray_x, axis=3)
			x = np.reshape(x, (-1, x.shape[0], x.shape[1], x.shape[2]))
			yield ({'input_1': gray_x/255.}, {'conv2d_24': x/255.})

def stepsOf(val):
	return ceil(len(val)/batch_size)



if __name__=='__main__':
	model = build_unet()
	#model.compile(optimizer=keras.optimizers.Adam(lr=.001), loss='mean_absolute_error', metrics=['accuracy'])
	print(model.summary())

	#model.fit(x=x_train/255., y=y_train/255., batch_size=batch_size, epochs=10, verbose=1,
	#	shuffle=True, validation_data=[x_test/255., y_test/255.])

	save_best = ModelCheckpoint('../weights/best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1, mode='min')
	checkpoint = ModelCheckpoint('../weights/chkpt_{epoch:04d}.h5', monitor='val_loss', save_best_only=False, verbose=1, mode='min', period=10)
	tensorboard = TensorBoard(log_dir='../logs/{}'.format(time.time()), batch_size=64)

	model.fit_generator(generator(x_train), steps_per_epoch=stepsOf(x_train), epochs=500, shuffle=True,
		validation_data=generator(x_test), validation_steps=stepsOf(x_test), callbacks=[save_best, checkpoint, tensorboard])
	model.save('../weights/colorizor.h5')

	acc = model.evaluate(x_test)
	print(acc[1])