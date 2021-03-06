from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from img_removemostcolor import simplify_img_random_vals
from mostly_gray_predict import show_output
import numpy as np
import matplotlib.pyplot as plt
def define_model(x_res, y_res):
	model = Sequential()
	model.add(Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',
					input_shape=(x_res, y_res, 3)))
	model.add(MaxPooling2D(pool_size = 2))
	model.add(Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(MaxPooling2D(pool_size = 2))
	model.add(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(UpSampling2D(size = (2,2)))
	model.add(Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(UpSampling2D(size = (2,2)))
	model.add(Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	#model.add(UpSampling2D(size = (2,2)))
	model.add(Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(Conv2D(3, 1, activation = 'sigmoid'))
	model.compile(optimizer = Adam(lr=1e-4, decay=1e-5), loss='mean_absolute_error')
	return model

def train(model, generator, n_epochs, batch_size, val_datagen=None, 
		initial_epoch=0, callbacks=None, multiprocessing=False):
	history = model.fit_generator(generator, steps_per_epoch=np.ceil(101000/batch_size), epochs=n_epochs,
						initial_epoch=initial_epoch, callbacks=callbacks, use_multiprocessing=multiprocessing,
						validation_data=val_datagen, validation_steps=np.ceil(101000/batch_size)//1e2)
	return history

def custom_generator(color_generator, grayscale_generator, debug=False):
	while True:
		x = grayscale_generator.next()
		y = color_generator.next()

		if debug:
			print('Grayscale shape:', x[0].shape)
			print('Color shape:', y[0].shape)
			show_output(x[0][23],y[0][23])
		yield (x[0],y[0])

if __name__ =='__main__':
	x_res, y_res = 256, 256
	model = define_model(x_res, y_res)
	batch_size = 32
	print(model.summary())
	datagen = ImageDataGenerator(preprocessing_function=simplify_img_random_vals,
									rescale=1/255.)
	datagen_y = ImageDataGenerator(rescale=1/255.)
	val_datagen = ImageDataGenerator(preprocessing_function=simplify_img_random_vals, rescale=1/255.)
	val_datagen_y = ImageDataGenerator(rescale=1/255.)
	generator = datagen.flow_from_directory('./imgs', target_size=(x_res, y_res), 
							seed=123, batch_size=batch_size)
	generator_y = datagen_y.flow_from_directory('./imgs', target_size=(x_res, y_res),
							seed=123, batch_size=batch_size)
	val_datagen = val_datagen.flow_from_directory('./imgs', target_size=(x_res, y_res),
						shuffle=True, seed=456, batch_size=batch_size)
	val_datagen_y = val_datagen_y.flow_from_directory('./imgs', target_size=(x_res, y_res),
						shuffle=True, seed=456, batch_size=batch_size)
	train_generator = custom_generator(generator, generator_y, debug=True)
	val_generator = custom_generator(val_datagen, val_datagen_y)
	ckpt = ModelCheckpoint('./weights/epoch{epoch:02d}.hdf5', save_best_only=False)
	tensorboard = TensorBoard(log_dir='./logs')


	history = train(model, train_generator, 3, batch_size, 
		val_datagen=val_generator, callbacks=[ckpt, tensorboard], initial_epoch=0)
	np.save('history', history)