import cv2
import numpy as np


def canny_edge_abs_pixels(img_array, min_val=100, max_val=200):
	transformed_array = []
	for img in img_array:
		h,w = img.shape
		edges = cv2.Canny(img, min_val, max_val)
		edge_pixel_count = sum(sum(edges))
		avg_edge_count = float(edge_pixel_count)/(h*w)
		transformed_array.append([avg_edge_count])
	return np.array(transformed_array)

def canny_edge_avg_pixels(img_array, min_val=100, max_val=200):
	transformed_array = []
	for img in img_array:
		edges = cv2.Canny(img, min_val, max_val)
		transformed_array.append([sum(sum(edges))])
	return np.array(transformed_array)


def only_spec_dim(img_array, x,y):
	transformed_array = []
	for img in img_array:
		if (x,y) == img.shape:
			transformed_array.append(img)
	return np.array(transformed_array)


def resize_all_images(img_array, size):
	return np.array([cv2.resize(img, size) for img in img_array])

def resize_image(img, size):
	return cv2.resize(img, size)


def print_avg_edge_pixels(img_path, min_val=100, max_val=200):
	img = cv2.imread(img_path, 0)
	edges = cv2.Canny(img, min_val, max_val)
	h,w = img.shape
	print 'Image shape:', h, 'x',w
	print 'Average Edge pixels:', round((sum(sum(edges))/float(h*w))*100, 2), '%'
	return None

