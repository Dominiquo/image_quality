import os
import cPickle
import numpy as np
import time
import random
from keras.preprocessing.image import DirectoryIterator, load_img, img_to_array
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


####################################### DATA IO ####################################################
def get_paths_labels(label_dir_dict):
	# TODO: HANDLE NO SUCH DIRECTORY EXCEPTION
	# TODO: HANDLE MORE THAN INTEGER LABELS
	max_label = max(label_dir_dict.keys())
	all_image_paths = []
	all_labels = []
	white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

	def in_white_list(fname, wlist):
			for suffix in wlist:
				if fname.lower().endswith(suffix): return True
			return False

	for label, dirs in label_dir_dict.iteritems():
		current_vecs = []
		for img_dir in dirs:
			current_vecs.extend([os.path.join(img_dir, p) for p in os.listdir(img_dir) if in_white_list(p, white_list_formats)])
		print 'Found', len(current_vecs), 'examples with label', label
		label_vec = np.array([label for l in xrange(len(current_vecs))])
		all_image_paths.extend(current_vecs)
		all_labels.extend(label_vec)

	return all_image_paths, np.array(all_labels, dtype='int32')


def load_data_by_labels(all_image_paths, all_labels, grayscale, target_size):
	all_feat_vecs = np.array([load_single_image_array(p, grayscale, target_size) for p in all_image_paths])
	print 'all feature vector shape', all_feat_vecs.shape
	return all_feat_vecs, all_labels


def load_data_by_directory(data_dir, grayscale, target_size):
	# ignore 'DS_Store' (problem with mac file storage)
	all_paths = [os.path.join(data_dir, p) for p in os.listdir(data_dir) if p != '.DS_Store']
	n_images = len(all_paths)
	if target_size != None:
		col_channels = 3
		x_len, y_len = target_size
		arr_shape = (n_images, x_len, y_len) if grayscale else (n_images, x_len, y_len, col_channels)
		all_images = np.empty(arr_shape)
		print 'Feature vectors shape:', all_images.shape, 'from directory:', data_dir
		start = time.time()
		for i, im_path in enumerate(all_paths):
			arr = load_single_image_array(im_path, grayscale, target_size).squeeze()
			all_images[i] = arr
		stop = time.time()
	else:
		start = time.time()
		all_images = []
		print 'loading images from', data_dir
		for i, im_path in enumerate(all_paths):
			arr = load_single_image_array(im_path, grayscale, None).squeeze()
			all_images.append(arr)
		all_images = np.array(all_images)
		stop = time.time()
	print 'Loaded', n_images, 'images in', round(stop - start, 2) , 'seconds.'
	return all_images


def shuffle_data(im_path_list, label_array):
	intermediate_shuffle = zip(im_path_list, label_array)
	random.shuffle(intermediate_shuffle)
	tup_path, tup_labels = zip(*intermediate_shuffle)
	return list(tup_path), np.array(tup_labels)


def load_batch_images(batch_filenames, grayscale, target_size):
	batch_size = len(batch_filenames)
	x,y = target_size
	if grayscale:
		image_batch = np.empty((batch_size, x, y))	
	else:
		c = 3
		image_batch = np.empty((batch_size, x, y, c))
	for i in xrange(batch_size):
		image_batch[i] = img_to_array(load_img(batch_filenames[i], grayscale=grayscale, target_size=target_size)).squeeze()
	return image_batch
	

def load_single_image_array(im_path, grayscale, target_size):
	return img_to_array(load_img(im_path, grayscale=grayscale, target_size=target_size))


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

