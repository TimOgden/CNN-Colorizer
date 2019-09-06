from keras.callbacks import Callback
import matplotlib.pyplot as plt
from img_simplifier import colormap
import cv2
import numpy as np

class OutputVisualizer(Callback):
	def __init__(self, image_url, save_ims=True):
		self.out_log = []
		self.image = cv2.resize(cv2.imread(image_url,0), dsize=(64,64), interpolation=cv2.INTER_CUBIC)
		self.color_image = cv2.cvtColor(cv2.resize(cv2.imread(image_url,1), dsize=(64,64), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
		self.save_ims = save_ims

	def on_epoch_end(self, epoch, logs={}):
		prediction = self.model.predict([self.image.reshape(-1,64,64,1), colormap(self.color_image).reshape(-1,64,64,3)], batch_size=2)
		self.out_log.append(prediction)
		if save_ims:
			cv2.imwrite('../logs/pictures/epoch{}gray.jpg'.format(epoch))
			cv2.imwrite('../logs/pictures/epoch{}predict.jpg'.format(epoch))
		plt.subplot(121)
		plt.imshow(self.image, cmap=plt.cm.gray)
		plt.title('Grayscale Image')
		plt.subplot(122)
		plt.imshow(prediction[0])
		plt.title('Neural Network Prediction')
		plt.show(block=False)
		plt.pause(2)
		plt.close()
