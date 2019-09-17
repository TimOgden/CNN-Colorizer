import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from outputvisualizer import OutputVisualizer
from img_simplifier import colormap
import numpy as np
from PIL import ImageFile
import matplotlib.pyplot as plt
import cv2
import time, datetime
import csv

ImageFile.LOAD_TRUNCATED_IMAGES = True

def gen_seed():
	seed = np.random.randint(0,999999999)
	seed_proven_unique = False
	while not seed_proven_unique:
		try:
			with open('seeds.txt', 'r') as f:
				seeds = csv.reader(f)
				#print(seeds)
				for val in seeds.__next__():
					#print(val)
					try:
						if int(val) == seed:
							seed = np.random.randint(0,999999999)
							continue
					except:
						pass
				seed_proven_unique = True
		except:
			seed_proven_unique = True
	return seed

def store_seed(seed, debug=False):
	with open('seeds.txt', 'a') as f:
		f.write('{},'.format(seed))
	if debug:
		print('Stored seed!')

def build_unet(pretrained_weights=None, input_size=(128,128)):
	input1 = Input(input_size + (1,))
	input2 = Input(input_size + (3,))

	inputs = concatenate([input1, input2], axis=-1)
	conv1 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv1a')(inputs)
	act1 = LeakyReLU()(conv1)
	conv2 = Conv2D(64, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv1b')(act1)
	act2 = LeakyReLU()(conv2)
	pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(act2)
	conv3 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv2a')(pool1)
	conv2 = Conv2D(128, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv2b')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
	conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv3a')(pool2)
	conv3 = Conv2D(256, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv3b')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)
	conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv4a')(pool3)
	conv4 = Conv2D(512, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv4b')(conv4)
	drop4 = Dropout(0.25, name='drop4')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(drop4)

	conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv5a')(pool4)
	conv5 = Conv2D(1024, 3, padding = 'same', kernel_initializer = 'he_normal', name='conv5b')(conv5)
	drop5 = Dropout(0.25, name='drop5')(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv6')(UpSampling2D(size = (2,2), name='up1')(drop5))
	merge6 = concatenate([drop4,up6], axis = -1)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv7a')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv7b')(conv6)

	up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
	merge7 = concatenate([conv3,up7], axis = -1)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
	merge8 = concatenate([conv2,up8], axis = -1)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
	merge9 = concatenate([conv1,up9], axis = -1)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

	model = keras.models.Model(input = [input1,input2], output = conv10)

	model.compile(optimizer = Adam(lr=1e-4, decay=1e-5), loss = 'mean_absolute_error')
	
	#model.summary()

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

def reg_model(pretrained_weights=None, input_size=(128,128)):
	input_a = Input(input_size + (1,))
	input_b = Input(input_size + (3,))

	# A
	conv1a = Conv2D(64, 1, strides=2, padding='same')(input_a)
	act1a = LeakyReLU()(conv1a)

	conv2a = Conv2D(128, 1, strides=2, padding='same')(act1a)
	act2a = LeakyReLU()(conv2a)

	# B
	conv1b = Conv2D(64, 3, strides=2, padding='same')(input_b)
	act1b = LeakyReLU()(conv1b)

	conv2b = Conv2D(128, 3, strides=2, padding='same')(act1b)
	act2b = LeakyReLU()(conv2b)

	combined = concatenate([act2a, act2b], axis=-1)

	# Combined A + B
	conv3 = Conv2D(64, 3, strides=1, padding='same')(combined)
	up3 = UpSampling2D()(conv3)
	act3 = LeakyReLU()(up3)

	conv4 = Conv2D(32, 3, strides=1, padding='same')(act3)
	up4 = UpSampling2D()(conv4)
	act4 = LeakyReLU()(up4)

	conv5 = Conv2D(16, 3, strides=1, padding='same')(act4)
	up5 = UpSampling2D()(conv5)
	act5 = LeakyReLU()(up5)

	conv6 = Conv2D(3, 1, activation='sigmoid', strides=2, padding='same')(act5)

	model = keras.models.Model(input = [input_a, input_b], output = conv6)
	model.compile(optimizer = Adam(lr=1e-3, decay=1e-5), loss = 'mean_absolute_error')
	if(pretrained_weights):
		model.load_weights(pretrained_weights)
	return model



def generate_generator_multiple(directory, generator, colormap_generator, batch_size, x_res, y_res, debug=False):
	train_x_gen = generator.flow_from_directory(directory,
													target_size=(x_res,y_res),
													batch_size=batch_size,
													class_mode=None,
													color_mode='grayscale',
													seed=1337,
													shuffle=True)
	
	train_y_gen = generator.flow_from_directory(directory,
													target_size=(x_res,y_res),
													batch_size=batch_size,
													class_mode=None,
													color_mode='rgb',
													seed=1337,
													shuffle=True)

	train_x_colormap_gen = colormap_generator.flow_from_directory(directory,
													target_size=(x_res,y_res),
													batch_size=batch_size,
													class_mode=None,
													color_mode='rgb',
													seed=1337,
													shuffle=True)
	while True:
			x = train_x_gen.next()
			x_colormap = train_x_colormap_gen.next()
			y = train_y_gen.next()
			if debug:
				plt.subplot(131)
				plt.imshow(x[0].reshape(64,64), cmap=plt.cm.gray)
				plt.title('Grayscale Image')
				plt.subplot(132)
				plt.imshow(x_colormap[0])
				plt.title('Simple Colormap')
				plt.subplot(133)
				plt.imshow(y[0])
				plt.title('Actual Colored Image')
				plt.show()
			yield [x, x_colormap], y  #Yield both images and their mutual label
			

if __name__ == '__main__':
	x_res, y_res = int(256/4), int(256/4)
	batch_size = 64
	seed = gen_seed()
	with tf.device('/gpu:0'):
		model = reg_model(input_size=(x_res,y_res))
		print(model.summary())
		datagen = ImageDataGenerator(dtype=np.uint8, rescale=1./255)
		colormap_datagen = ImageDataGenerator(preprocessing_function=colormap, dtype=np.uint8, rescale=1./255)
		
		train_generator = generate_generator_multiple('D:/imagenet_train/', datagen, colormap_datagen,
														batch_size, x_res, y_res, debug=False)
		print('Created generator!')
		val_generator = generate_generator_multiple('D:/imagenet_val/', datagen, colormap_datagen,
														batch_size, x_res, y_res)
		print('Created validation generator!')
		store_seed(seed)
		model.fit_generator(train_generator, validation_data=val_generator, epochs=10,
			steps_per_epoch=np.ceil(1e5/(batch_size*2)), validation_steps=np.ceil(2e4/(batch_size*2)),
			callbacks=[ TensorBoard(log_dir='../logs', batch_size=batch_size),
						EarlyStopping(patience=1, restore_best_weights=True),
						ModelCheckpoint('../weights/epoch{epoch:02d}-%s.h5' % (seed),
								save_best_only=True, verbose=1, save_weights_only=True)])
		