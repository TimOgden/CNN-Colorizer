import cv2
import matplotlib.pyplot as plt
import numpy as np
from colorizor import build_unet
import time

def cvt_gray(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	return np.reshape(gray, (1, 32, 32, 1))/255.
# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera
model = build_unet(pretrained_weights = '../weights/best_0.6.h5')

def one_img():
	s, img = cam.read()

	if s:    # frame captured without any errors
		img = img[60:316, 100:356]
		img = cv2.resize(img, (32,32))
		#plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		#plt.show()
		#cv2.imwrite("filename.jpg",img) #save image

		
		yhat = model.predict(cvt_gray(img))
		plt.subplot(1,3,1)
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

		plt.subplot(1,3,2)
		plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap='gray')

		yhat = np.reshape(yhat, (32,32,3))
		plt.subplot(1,3,3)
		plt.imshow(yhat)

		plt.show()

def stream(recordingTime = 10):
	# Check if camera opened successfully
	if (cam.isOpened() == False): 
		print("Unable to read camera feed")
	 
	# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
	out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (32,32))
	orig = cv2.VideoWriter('orig.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (32,32))
	start = time.time()

	while(True):
		ret, frame = cam.read()
		if ret == True:
			frame = frame[60:316, 100:356]
			frame = cv2.resize(frame, (32,32))
			orig.write(frame)
			# Write the frame into the file 'output.avi'
			frame = cvt_gray(frame)
			yhat = model.predict(frame)
			yhat = np.uint8(255 * yhat)
			yhat = np.reshape(yhat, (32, 32, 3))
			out.write(yhat)
		
			# Display the resulting frame
		
			# Press Q on keyboard to stop recording
			if time.time() - start >= recordingTime:
				break
	
		# Break the loop
		else:
			break 
	# When everything done, release the video capture and video write objects
	cam.release()
	out.release()
	 
	# Closes all the frames
	cv2.destroyAllWindows()

stream()