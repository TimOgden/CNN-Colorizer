from keras.callbacks import Callback
import matplotlib.pyplot as plt
from img_simplifier import colormap
import cv2
import numpy as np
from os import walk

class OutputVisualizer(Callback):
	def __init__(self, folder_url, save_ims=True, time_to_display_ims=-1):
		self.out_log = []
		self.folder_url = folder_url
		(_, _, filenames) = walk(folder_url).__next__()
		filenames = [folder_url + f for f in filenames]
		#print(filenames)
		self.images = []
		self.colormaps = []
		for f in filenames:
			self.images.append(cv2.resize(cv2.imread(f,0), dsize=(64,64), interpolation=cv2.INTER_CUBIC))
			self.colormaps.append(cv2.cvtColor(cv2.resize(cv2.imread(f,1), dsize=(64,64), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
		self.save_ims = save_ims
		self.time_to_display_ims = time_to_display_ims

		self.images = np.divide(self.images,255.).reshape(-1,64,64,1)
		self.colormaps = [colormap(img) for img in self.colormaps]
		self.colormaps = np.divide(self.colormaps,255.).reshape(-1,64,64,3)

	def on_epoch_end(self, epoch, logs={}):
		predictions = self.model.predict([self.images,self.colormaps], batch_size=2)
		self.out_log.append(predictions)
		if self.save_ims:
			c = 0
			for pred in predictions:
				cv2.imwrite('../logs/pictures/epoch{}/gray{}.jpg'.format(epoch+1,c+1), np.multiply(self.images[c],255.))
				cv2.imwrite('../logs/pictures/epoch{}/predict{}.jpg'.format(epoch+1,c+1), np.multiply(pred,255.))
				c+=1
		if self.time_to_display_ims>0:
			plt.subplot(121)
			plt.imshow(self.images[0].reshape(64,64), cmap=plt.cm.gray)
			plt.title('Grayscale Image')
			plt.subplot(122)
			plt.imshow(predictions[0])
			plt.title('Neural Network Prediction')
			plt.show(block=False)
			plt.pause(self.time_to_display_ims)
			plt.close()