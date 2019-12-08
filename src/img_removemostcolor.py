import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools

def simplify_img_random_vals(img, grid_size=4):
	''' Quick and easy function to simplify the img with random parameters.
		Combination of simplify_img and draw_squares

	Keyword arguments:
	img (numpy.ndarray) - input color image
	grid_size (int) - the size of each square in the grid
	'''
	percent_in_color = np.random.uniform(low=.01,high=.07)
	square_size = int(np.random.uniform(low=2,high=16))
	squares = simplify_img(img, percent_in_color=percent_in_color, grid_size=grid_size)
	return draw_squares(img, squares, square_size=square_size)


def simplify_img(img, percent_in_color=.1, grid_size=32):
	''' Divides the image into a grid, chooses a percentage of the squares and gets the
	color of the center pixel of each square.

	Keyword arguments:
	img (numpy.ndarray) - input color image
	percent_in_color (float) - percentage of the squares to color
	grid_size - the size of each square in the grid (not necessarily the same as square_size in draw_squares)
	'''
	# Put the image into a grid
	num_squares_per_row = img.shape[0] // grid_size
	num_squares_per_col = img.shape[1] // grid_size
	squares_count = int(np.floor(num_squares_per_row * num_squares_per_col))
	squares = list(itertools.product(*[range(num_squares_per_row),range(num_squares_per_col)]))
	
	# Choose a number of those squares to be colored
	num_to_choose = int(len(squares)*percent_in_color)
	
	chosen_squares = list(squares[i] for i in np.random.permutation(len(squares))[:num_to_choose])
	
	# Get the average color of the squares
	grid_colors = {}
	for square in chosen_squares:
		x = square[0]*grid_size+grid_size//2
		y = square[1]*grid_size+grid_size//2
		color = img[x][y]
		grid_colors[(x,y)] = color
		
	return grid_colors


def draw_squares(img, square, square_size=5):
	''' Draws all the squares in the square dictionary with given color

	Keyword arguments:
	square (dict) - dictionary containing the squares (with x,y as keys and color as values)
	square_size (int) - size of the squares when drawing
	'''
	
	square_size//=2
	for (x,y) in square.keys():
		# Need two for loops for each square
		for r in range(x-square_size, x+square_size):
			for c in range(y-square_size, y+square_size):
				try:
					img[r][c] = square[(x,y)] # Get the color associated with the square
				except:
					pass # In case the coordinate is not valid
	return img

if __name__ =='__main__':
	img = cv2.cvtColor(cv2.imread('oasis.png'), cv2.COLOR_BGR2RGB)
	squares = simplify_img(img, percent_in_color=.01)
	grayscale_3channels = cv2.cvtColor(cv2.imread('oasis.png', 0), 
								cv2.COLOR_GRAY2RGB)
	print(grayscale_3channels.shape)
	gray_img = draw_squares(grayscale_3channels, squares, square_size=16)
	
	plt.subplot(1,2,1)
	plt.title('Original Image')
	plt.imshow(img)
	plt.subplot(1,2,2)
	plt.title('Mostly Grayscale Image')
	plt.imshow(gray_img)
	plt.show()