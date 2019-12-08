from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
from img_removemostcolor import simplify_img_random_vals
import numpy as np

def define_model(x_res, y_res):
	model = Sequential()
	model.add(Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',
					input_shape=(x_res, y_res, 3)))
	model.add(Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal'))
	model.add(Dense(3, activation = 'sigmoid'))
	model.compile(optimizer = Adam(lr=1e-4, decay=1e-5), loss='mean_squared_error')
	return model

def train(model, generator, n_epochs, batch_size, initial_epoch=0, callbacks=None):
	model.fit_generator(generator, steps_per_epoch=np.ceil(1e6/batch_size), epochs=n_epochs,
						initial_epoch=initial_epoch, callbacks=callbacks)

if __name__ =='__main__':
	model = define_model(64,64)
	datagen = ImageDataGenerator(horizontal_flip=True, preprocessing_function=simplify_img_random_vals,
									rescale=1/255., validation_split=.2)
	generator = datagen.flow_from_directory('./imgs', target_size=(64,64))
	ckpt = ModelCheckpoint('./weights/epoch{epoch:02d}.hdf5', save_best_only=True)
	tensorboard = TensorBoard(log_dir='./logs')
	train(model, generator, 10, 32, callbacks=[ckpt, tensorboard])
