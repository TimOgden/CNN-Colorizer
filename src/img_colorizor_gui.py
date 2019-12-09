from tkinter import *
from tkinter.colorchooser import askcolor
from PIL import ImageTk,Image
import cv2
import matplotlib.pyplot as plt
import numpy as np

class GUI:
	def __init__(self):
		self.on_click_x = 0
		self.off_click_x = 0
		self.on_click_y = 0
		self.off_click_y = 0
		self.color = None
		self.image_list = []
		self.np_image = cv2.cvtColor(cv2.imread('oasis.png',0), cv2.COLOR_BGR2RGB)
		self.image_list.append(np.copy(self.np_image))
		self.root = Tk()
		self.canvas = Canvas(self.root, width = 256, height = 256)
		self.canvas.bind('<Button-1>', self.click_press)
		self.canvas.bind('<ButtonRelease-1>', self.click_release)
		self.canvas.pack()
		
		print(self.np_image.shape)
		self.img = ImageTk.PhotoImage(Image.fromarray(self.np_image))
		self.canvas.create_image(0,0, anchor=NW, image=self.img)
		self.buttom_img = ImageTk.PhotoImage(Image.fromarray(cv2.imread('undo.png')))
		b = Button(self.root, height=40, width=40, image=self.buttom_img,
					command=self.undo, anchor=SW).pack()

	def undo(self):
		print('Trying to undo')
		try:
			self.image_list.pop()
			print(len(self.image_list))
			self.np_image = np.copy(self.image_list[-1])
			#plt.imshow(self.np_image)
			#plt.show()
			redraw_img(self.np_image)
		except:
			pass

	def set_color(self, debug=False):
		color = askcolor()

		if debug:
			print(color)
		return color

	def click_press(self, event):
		self.on_click_x = event.x
		self.on_click_y = event.y

	def click_release(self, event):
		self.off_click_x = event.x
		self.off_click_y = event.y
		color = self.set_color()
		self.draw_square(color)
		print(self.np_image.shape)

	def draw_square(self, color):
		temp = 0
		
		if self.on_click_x > self.off_click_x:
			temp = self.on_click_x
			self.on_click_x = self.off_click_x
			self.off_click_x = temp
		if self.on_click_y > self.off_click_y:
			temp = self.off_click_y
			self.on_click_y = self.off_click_y
			self.off_click_y = temp
		for x in range(self.on_click_x, self.off_click_x):
			for y in range(self.on_click_y, self.off_click_y):
				#print('Numpy image', self.np_image is not None)
				#print('Color', self.color is not None)
				self.np_image[y][x] = color[0]
		self.image_list.append(np.copy(self.np_image))
		self.redraw_img(self.np_image)

	def redraw_img(self, image):
		print('Redrawing image')
		self.img = ImageTk.PhotoImage(Image.fromarray(image))
		
		#self.canvas.configure(image=self.img)
		self.canvas.create_image(0,0, anchor=NW, image=self.img)
		self.root.update_idletasks()
		#plt.imshow(image)
		#plt.show()



if __name__=='__main__':
	gui = GUI()
	gui.root.mainloop()