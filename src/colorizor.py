from keras.datasets import cifar10
from keras.layers import Conv2D, UpSampling2D, Flatten, Dense, MaxPooling2D, Input
from keras.optimizers import Adam
from keras.layers import LeakyReLU, Dropout
from keras.models import Sequential
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
			yield ({'conv2d_1_input': gray_x}, {'conv2d_3': x})

def stepsOf(val):
	return ceil(len(val)/batch_size)

model = build_model()
model.compile(optimizer=keras.optimizers.Adam(lr=.0001), loss='mean_absolute_error', metrics=['accuracy'])
print(model.summary())

#model.fit(x=x_train/255., y=y_train/255., batch_size=batch_size, epochs=10, verbose=1,
#	shuffle=True, validation_data=[x_test/255., y_test/255.])

save_best = ModelCheckpoint('../weights/best.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, verbose=1, mode='min')
checkpoint = ModelCheckpoint('../weights/chkpt_{epoch:04d}.h5', monitor='val_loss', save_best_only=False, verbose=1, mode='min', period=5)
tensorboard = TensorBoard(log_dir='../logs/{}'.format(time.time()), batch_size=64)

model.fit_generator(generator(x_train), steps_per_epoch=stepsOf(x_train), epochs=500, shuffle=True,
	validation_data=generator(x_test), validation_steps=stepsOf(x_test), callbacks=[save_best, checkpoint, tensorboard])
model.save('../weights/colorizor.h5')

acc = model.evaluate(x_test)
print(acc[1])