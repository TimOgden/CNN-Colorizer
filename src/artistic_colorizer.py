import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from img_simplifier import colormap
import numpy as np

def build_unet(pretrained_weights=None, input_size=(128,128)):
	input1 = Input(input_size + (1,))
	input2 = Input(input_size + (3,))

	inputs = concatenate([input1, input2], axis=-1)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1a')(inputs)
	conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv1b')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2a')(pool1)
	conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv2b')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv3a')(pool2)
	conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv3b')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4a')(pool3)
	conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv4b')(conv4)
	drop4 = Dropout(0.5, name='drop4')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(drop4)

	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5a')(pool4)
	conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv5b')(conv5)
	drop5 = Dropout(0.5, name='drop5')(conv5)

	up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv6')(UpSampling2D(size = (2,2), name='up1')(drop5))
	merge6 = concatenate([drop4,up6], axis = 3)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv7a')(merge6)
	conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='conv7b')(conv6)

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
	conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

	model = keras.models.Model(input = [input1,input2], output = conv10)

	model.compile(optimizer = Adam(lr=1e-4, decay=1e-5), loss = 'mean_squared_error')
	
	#model.summary()

	if(pretrained_weights):
		model.load_weights(pretrained_weights)

	return model

def generate_generator_multiple(generator, colormap_generator, batch_size, x_res, y_res):
	train_x_gen = generator.flow_from_directory('D:/imagenet_train/ILSVRC2014_DET_train',
													target_size=(x_res,y_res),
													batch_size=batch_size,
													class_mode=None,
													color_mode='grayscale',
													seed=1,
													shuffle=True)
	
	train_y_gen = generator.flow_from_directory('D:/imagenet_train/ILSVRC2014_DET_train',
													target_size=(x_res,y_res),
													batch_size=batch_size,
													class_mode=None,
													color_mode='rgb',
													seed=1,
													shuffle=True)

	train_x_colormap_gen = colormap_generator.flow_from_directory('D:/imagenet_train/ILSVRC2014_DET_train',
													target_size=(x_res,y_res),
													batch_size=batch_size,
													class_mode=None,
													color_mode='rgb',
													seed=1,
													shuffle=True)
	while True:
			x = train_x_gen.next()
			x_colormap = train_x_colormap_gen.next()
			y = train_y_gen.next()
			yield [x, x_colormap], y  #Yield both images and their mutual label

if __name__ == '__main__':
	x_res, y_res = int(256/2), int(256/2)
	batch_size = 7
	model = build_unet()
	print(model.summary())
	train_datagen = ImageDataGenerator(dtype=np.uint8)
	train_colormap_datagen = ImageDataGenerator(preprocessing_function=colormap, dtype=np.uint8)
	
	generator = generate_generator_multiple(train_datagen, train_colormap_datagen, batch_size, x_res, y_res)
	print('Created generator!')

	model.fit_generator(generator, epochs=10, steps_per_epoch=np.ceil(107505/batch_size),
		callbacks=[ TensorBoard(log_dir='./logs', batch_size=batch_size),
					EarlyStopping(patience=2),
					ModelCheckpoint('model.h5', save_best_only=True, verbose=1)])