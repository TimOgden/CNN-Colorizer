import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
def simplify_img(img, percent_in_color=.1, square_size=32):
	''' Divides the image into a grid, chooses a percentage of the squares and gets the
	color of the center pixel of each square.

	Keyword arguments:
	img (numpy.ndarray) - input color image
	percent_in_color (float) - percentage of the squares to color
	square_size - the size of each square in the grid (not necessarily the same as square_size in draw_squares)
	'''
	# Put the image into a grid
	num_squares_per_row = img.shape[0] // square_size
	num_squares_per_col = img.shape[1] // square_size
	squares_count = int(np.floor(num_squares_per_row * num_squares_per_col))
	squares = list(itertools.product(*[range(num_squares_per_row),range(num_squares_per_col)]))
	
	# Choose a number of those squares to be colored
	num_to_choose = int(len(squares)*percent_in_color)
	
	chosen_squares = list(squares[i] for i in np.random.permutation(len(squares))[:num_to_choose])
	
	# Get the average color of the squares
	grid_colors = {}
	for square in chosen_squares:
		print('Square[0]:', square[0],'Square[1]:', square[1],'square_size:', square_size)
		x = square[0]*square_size+square_size//2
		y = square[1]*square_size+square_size//2
		color = img[x][y]
		grid_colors[(x,y)] = color
		print(color)
	return grid_colors


def draw_squares(img, square, square_size=5):
	''' Draws all the squares in the square dictionary with given color

	Keyword arguments:
	square (dict) - dictionary containing the squares (with x,y as keys and color as values)
	square_size (int) - size of the squares when drawing
	'''
	print('Image shape:', img.shape)
	square_size//=2
	for (x,y) in square.keys():
		# Need two for loops for each square
		for r in range(x-square_size, x+square_size):
			for c in range(y-square_size, y+square_size):
				img[r][c] = square[(x,y)] # Get the color associated with the square
				#img[r][c] = square[(x,y)][1]
				#img[r][c] = square[(x,y)][2]
	return img

if __name__ =='__main__':
	img = cv2.cvtColor(cv2.imread('oasis.png'), cv2.COLOR_BGR2RGB)
	squares = simplify_img(img, percent_in_color=.01)
	grayscale_3channels = cv2.cvtColor(cv2.imread('oasis.png', 0), 
								cv2.COLOR_BGR2RGB)
	print(grayscale_3channels.shape)
	gray_img = draw_squares(grayscale_3channels, squares, square_size=16)
	#gray_img = np.swapaxes(np.swapaxes(gray_img,0,1), 1,2) # flip from channels_first to channels_last
	plt.subplot(2,1,1)
	plt.imshow(img)
	plt.subplot(2,1,2)
	plt.imshow(gray_img)
	plt.show()