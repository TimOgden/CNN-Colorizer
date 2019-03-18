import cv2
import matplotlib.pyplot as plt
import numpy as np
from colorizor import build_unet


def cvt_gray(img):
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	return np.reshape(gray, (1, 32, 32, 1))/255.
# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()

if s:    # frame captured without any errors
	img = img[60:316, 100:356]
	img = cv2.resize(img, (32,32))
	#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	#plt.show()
	#cv2.imwrite("filename.jpg",img) #save image

	model = build_unet(pretrained_weights = '../weights/best_0.6.h5')
	yhat = model.predict(cvt_gray(img))
	plt.subplot(1,3,1)
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

	plt.subplot(1,3,2)
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')

	yhat = np.reshape(yhat, (32,32,3))
	plt.subplot(1,3,3)
	plt.imshow(yhat)

	plt.show()
