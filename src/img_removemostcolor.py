import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
def simplify_img(img, percent_in_color=.1, square_size=32):
	# Put the image into a grid
	num_squares_per_row = img.shape[0] // square_size
	num_squares_per_col = img.shape[1] // square_size
	print(num_squares_per_row, num_squares_per_col)
	squares_count = int(np.floor(num_squares_per_row * num_squares_per_col))
	squares = list(itertools.product(*[range(num_squares_per_row),range(num_squares_per_col)]))
	
	# Choose a number of those squares to be colored
	print(int(len(squares)))
	num_to_choose = int(len(squares)*percent_in_color)
	print('Number to choose:', num_to_choose)
	chosen_squares = list(squares[i] for i in np.random.permutation(len(squares))[:num_to_choose])
	print(chosen_squares, len(chosen_squares))
	# Get the average color of the squares
	for square in chosen_squares:
		print('Square[0]:', square[0],'Square[1]:', square[1],'square_size:', square_size)
		color = img[square[0]*square_size+square_size//2][square[1]*square_size+square_size//2]
		print(color)



def draw_square(img, square):
	print(type(square[0][0]), type(square[2]))
	for r in range(square[0][0] - square[2], square[0][0] + square[2]):
		for c in range(square[0][1] - square[2], square[0][1] + square[2]):
			img[r][c] = square[1]
	return img

if __name__ =='__main__':
	img = cv2.imread('oasis.png')
	simplify_img(img, percent_in_color=.01)
	#grayscale_3channels = np.tile(cv2.imread('oasis.png', 0), (1,1,3))
	#for square in squares:
	#	gray_img = draw_square(grayscale_3channels, square)
	#plt.subplot(2,1,1)
	#plt.imshow(img)
	#plt.subplot(2,1,2)
	#plt.imshow(gray_img)
	#plt.show()