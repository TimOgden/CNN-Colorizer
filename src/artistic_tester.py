import keras
from img_simplifier import colormap
import artistic_colorizer
from cv2 import imread
import matplotlib.pyplot as plt
def reshape_img(img):
	img = img / 255.
	return np.reshape(img, (-1, img.shape[0], img.shape[1], 3))

if __name__=='__main__':
	#model = artistic_colorizer.reg_model(input_size=(64,64))
	
	model = keras.models.load_model('../weights/best.h5')
	x = reshape_img(imread('D:/imagenet_val/ILSVRC2013_DET_val/ILSVRC2012_val_00000001.jpg', 0))
	x_real = reshape_img(imread('D:/imagenet_val/ILSVRC2013_DET_val/ILSVRC2012_val_00000001.jpg', 1))
	x_colormap = colormap(x)
	x_test = [[x,x_colormap]]
	y_pred = model.predict(x_test)[0]

	plt.subplot(141)
	plt.imshow(x)
	plt.title('Grayscale Image')
	plt.subplot(142)
	plt.imshow(x_colormap)
	plt.title('Simple Colormap')
	plt.subplot(143)
	plt.imshow(y_pred)
	plt.title('Neural Network Prediction')
	plt.subplot(144)
	plt.imshow(x_real)
	plt.title('Actual Colored Image')
	plt.show()
