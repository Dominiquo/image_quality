import cv2
import os
import cPickle
import numpy as np


def load_data(data_dir, setting=1):
	# ignore 'DS_Store' (problem with mac file storage)
	all_paths = [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p != '.DS_Store']
	print 'loading all files from', data_dir, '...'
	try:
		all_images = [cv2.imread(p, setting) for p in all_paths]
	except Exception as e:
		print e
		return None
	return np.array(all_images)


def load_object(stored_path):
	with open(stored_path) as infile:
		print 'retrieving object from', stored_path, '...'
		obj = cPickle.load(infile)
	return obj


def store_object(obj, store_path):
	with open(store_path, 'wb') as outfile:
		print 'stored object at', store_path
		cPickle.dump(obj, outfile)
	return True


def apply_to_images(image_dir):
	file_count = 0
	edge_count = 0
	print 'processing images from path:', image_dir, '...'
	all_images = os.listdir(image_dir)
	total_images = len(all_images)
	for image_name in all_images:
		# cavaet code because mac likes to add '.DS_Store' to everything
		if image_name[0] == '.': continue
		image_path = os.path.join(image_dir, image_name)
		edge_count += sum_edges(image_path)
		file_count += 1
		if file_count % 100 == 0:
			print round((float(file_count)/total_images)*100, 2), '%', 'complete'
	return float(edge_count)/file_count

