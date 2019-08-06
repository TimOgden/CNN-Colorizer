import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np

def convert_im(image, show_img=False, save_img=False):
	orig_img = cv2.imread(image)
	img = cv2.GaussianBlur(orig_img, (5,5), 0)
	img = cv2.pyrMeanShiftFiltering(img, 20, 45, 3)

	if show_img:
		plt.title('Output')
		plt.subplot(121)
		plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
		plt.title('Original')
		plt.subplot(122)
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		plt.title('New')
		plt.show()
	if save_img:
		cv2.imwrite('output.png', img)

	return img

def colormap(img):
	print(img.dtype)
	img = cv2.GaussianBlur(np.int8(img), (5,5), 0)
	return cv2.pyrMeanShiftFiltering(img, 20, 45, 3)

if __name__=='__main__':
	print(sys.argv[1])
	convert_im(sys.argv[1], show_img=True, save_img=True)