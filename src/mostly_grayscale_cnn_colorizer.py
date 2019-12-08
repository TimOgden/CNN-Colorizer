from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from img_removemostcolor import simplify_img_random_vals
import numpy as np

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
	model.compile(optimizer = Adam(lr=1e-4, decay=1e-5), loss='mean_squared_error')
	return model

def train(model, generator, n_epochs, batch_size, initial_epoch=0, callbacks=None):
	model.fit_generator(generator, steps_per_epoch=np.ceil(101000/batch_size), epochs=n_epochs,
						initial_epoch=initial_epoch, callbacks=callbacks)

def custom_generator(color_generator, grayscale_generator):
	while True:
		x = color_generator.next()
		y = grayscale_generator.next()
		yield (x[0],y[0])

if __name__ =='__main__':
	model = define_model(64,64)
	batch_size = 32
	print(model.summary())
	datagen = ImageDataGenerator(horizontal_flip=True, preprocessing_function=simplify_img_random_vals,
									rescale=1/255., validation_split=.2)
	datagen_y = ImageDataGenerator(horizontal_flip=True, rescale=1/255., validation_split=.2)
	generator = datagen.flow_from_directory('./imgs', target_size=(64,64), seed=123, batch_size=batch_size)
	generator_y = datagen_y.flow_from_directory('./imgs', target_size=(64,64), seed=123, batch_size=batch_size)
	train_generator = custom_generator(generator, generator_y)
	ckpt = ModelCheckpoint('./weights/epoch{epoch:02d}.hdf5', save_best_only=True)
	tensorboard = TensorBoard(log_dir='./logs')


	train(model, train_generator, 10, batch_size, callbacks=[ckpt, tensorboard])
