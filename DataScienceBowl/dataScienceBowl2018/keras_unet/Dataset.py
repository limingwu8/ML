
# coding: utf-8

# In[9]:

import os
import re
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.misc import imfilter

class Dataset():

	img_w = 256
	img_h = 256

	def __init__(self, img_w, img_h, file_dir):
		self.img_w = img_w
		self.img_h = img_h
		self.file_dir = file_dir
		self.img_dim = []

	def get_masks(self, paths):
		"""
		overlap these mask images into one mask image
		:param temp: a bounch of images given in file path
		:return: the overlaped image in numpy array
		"""
		image = Image.open(paths[0]).resize((self.img_w,self.img_h))
		for path in paths:
			img = Image.open(path)
			img = img.resize((self.img_w,self.img_h))
			image = np.array(image) | np.array(img)

		return image.reshape((self.img_w, self.img_h, 1))

	def get_contours(self, images, labels):
		'''
		get the contours of the label, and map the contour to image and label
		:param images: images, ndarray, e.g.(500,256,256,4)
		:param labels: masks, ndarray, e.g.(500,256,256,1)
		:return:
		images with contour, e.g.(500,256,256,4)
		labels with contour, e.g.(500,256,256,1)
		contours, e.g.(500,256,256,1)
		'''
		num_imgs = images.shape[0]

		contour_images = []
		contour_labels = []
		contours = []
		for i in range(0, num_imgs):
			image = images[i]
			label = labels[i]
			contour = imfilter(label.reshape(label.shape[0], label.shape[1]), "find_edges")
			contour = contour.reshape((contour.shape[0], contour.shape[1], 1))
			contour_images.append(image | contour)
			contour_labels.append(label | contour)
			contours.append(contour)
		# self.contour_images = contour_images
		# self.contour_labels = contour_labels
		# self.contours = contours
		return contour_images, contour_labels, contours

	def plot_images(self, images, labels, contour_images, contours, index = -1):
		if index == -1:
			index = np.random.randint(images.shape[0])
		image = images[index]
		label = labels[index].reshape(self.img_w, self.img_h)
		contour_image = contour_images[index]
		contour = contours[index].reshape(self.img_w, self.img_h)
		imgs = [image,label,contour_image,contour]
		titles = ['image','label','contour image', 'contour']
		# fig, axes = plt.subplot(2,2)
		for i in range(0, len(imgs)):
			plt.subplot(2, 2, i+1)
			plt.title(titles[i])
			plt.subplots_adjust(top = 0.96, bottom = 0.02, left = 0.1, right = 0.9, hspace = 0.1, wspace = 0)
			plt.imshow(imgs[i])
		plt.show()

	def read_files(self, file_dir):
		'''
		Args:
			file_dir: root file directory
		Returns:
			training images, in ndarray, e.g. (500,256,256,4)
			training labels, in ndarray, e.g. (500,256,256,1)
		'''
		images = []
		temp = []
		labels = []
		for root, dirs, files in os.walk(file_dir):
			parent_dir_name = re.split(r'/',root)[-1]
			# image directories
			for file in files:
				temp.append(os.path.join(root, file))

			if parent_dir_name == 'masks':
				labels.append(self.get_masks(temp))
				temp.clear()
			elif parent_dir_name == 'images':
				img = Image.open(os.path.join(root, file))
				# img = img.convert("L")
				self.img_dim.append((img.size))
				img = img.resize((self.img_w,self.img_h))
				images.append(np.array(img))
				temp.clear()

		images = np.array(images)
		labels = np.array(labels)
		# self.images = images
		# self.labels = labels
		return images, labels

	def prepare(self):
		images, labels = self.read_files(self.file_dir)
		contour_images, contour_labels, contours = self.get_contours(images, labels)
		self.contour_images = contour_images
		self.contour_labels = contour_labels
		self.contours = contours
		self.images = images
		self.labels = labels

		return images, labels, contour_images, contour_labels, contours

if __name__ == '__main__':
	img_w = img_h = 256
	file_dir = '/root/dataset/dataScienceBowl2018/stage1_train'
	dataset = Dataset(img_w,img_h,file_dir)

	images, labels, contour_images, contours = dataset.prepare()

	dataset.plot_images(images, labels, contour_images,contours)


	plt.pause()



