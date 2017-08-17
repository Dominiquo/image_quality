import os
import cPickle
import cv2
import numpy as np
import copy
import time
import random
from misc import utils
from keras.preprocessing.image import load_img, img_to_array, random_rotation, random_shear, random_zoom, flip_axis


# APPLIED FUNCTIONS: RESCALE, SHEAR_RANGE, ZOOM_RANGE, HORIZONTAL_FLIP


class DataGen(object):
	def __init__(self, im_paths, labels, batch_size, shuffle, grayscale, target_size):
		self.batch_size = batch_size
		self.total_batches = len(im_paths)/self.batch_size
		self.total_values = self.total_batches * self.batch_size
		self.paths = im_paths[:self.total_values]
		self.labels = labels[:self.total_values]
		self._paths = copy.deepcopy(self.paths)
		self._labels = copy.deepcopy(self.labels)
		self.shuffle = shuffle
		self.grayscale = grayscale
		self.target_size = target_size
		self.batch_index = 0

		self.use_transformations = False
		self.rot_range = 0
		self.shear_range = 0
		self.zoom_range = 0

	def assign_transformations(self, rot_range, shear_range, zoom_range):
		self.use_transformations = True
		self.rot_range = rot_range
		self.shear_range = shear_range
		self.zoom_range = zoom_range
		
	def __iter__(self): 
		return self

	def next(self):
		if self.shuffle:
			self._paths, self._labels = utils.shuffle_data(self.paths, self.labels)

		while(True):
			lower = (self.batch_index % self.total_batches)*self.batch_size
			upper = (self.batch_index % self.total_batches)*self.batch_size + self.batch_size
			batch_imgs = utils.load_batch_images(self._paths[lower:upper], self.grayscale, self.target_size)

			if self.use_transformations:
				for i,im in enumerate(batch_imgs):
					batch_imgs[i] = image_random_transform(im, self.rot_range, self.shear_range, self.zoom_range)

			batch_labels = self._labels[lower:upper]
			self.batch_index += 1
			if ((self.batch_index % self.total_batches) == 0) and (self.shuffle):
				self._paths, self._labels = utils.shuffle_data(self._paths, self._labels)

			return batch_imgs, batch_labels

	def reset(self):
		self.batch_index = 0
		return None


def image_random_transform(img, rot_range, shear_range, zoom_range):
	zoom_val = (1-zoom_range, 1-zoom_range)
	new_img = random_rotation(img, rot_range, row_axis=0, col_axis=1, channel_axis=2)
	new_img = random_shear(new_img, shear_range, row_axis=0, col_axis=1, channel_axis=2)
	new_img = random_zoom(new_img, zoom_val, row_axis=0, col_axis=1, channel_axis=2)
	return new_img
