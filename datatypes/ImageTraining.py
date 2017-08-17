import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from misc import utils
from datatypes import DataGenerator as dg
from datatypes import ImageGenerator as ig


class ImageTraining(object):
	def __init__(self, labels_dir_dict):
		# TODO: add check that labels dir is correct format, throw exception
		paths, labels = utils.get_paths_labels(labels_dir_dict)
		self.labels_dirs = labels_dir_dict
		self.all_paths = paths
		self.all_labels = labels
		self.n_classes = max(labels_dir_dict.keys()) + 1
		self.input_shape = None
		self.X = None
		self.y = None
		self.X_train = None
		self.X_test = None
		self.y_train = None
		self.y_test = None


	def get_vecs_labels(self):
		if (self.X != None) and (self.y != None):
			return self.X, self.y
		else:
			raise Exception("Items not yet initialized.")


	def get_train_test_vecs(self):
		all_vals = [self.X_train, self.X_test, self.y_train, self.y_test]
		if all(v is not None for v in all_vals):
			return self.X_train, self.X_test, self.y_train, self.y_test
		else:
			raise Exception("Items not yet initialized.")


	def create_vectors(self, grayscale=False, target_size=None):
		self.X, self.y = utils.load_data_by_labels(self.all_paths, self.all_labels, grayscale, target_size)
		return self.X, self.y


	def create_test_train_vectors(self, train_size=.8, grayscale=False, target_size=(150,150), random_state=42):
		path_train, path_test, label_train, label_test = partition_train_test(
															self.all_paths, self.all_labels, train_size, random_state)
		self.X_train, self.y_train = utils.load_data_by_labels(path_train, label_train, grayscale, target_size)
		self.X_test, self.y_test = utils.load_data_by_labels(path_test, label_test, grayscale, target_size)
		return self.X_train, self.X_test, self.y_train, self.y_test


	def get_all_data_generator(self,  train_size=.8, batch_size=16, shuffle=True, grayscale=False, target_size=(150,150), random_state=42):
		datagen = ImageDataGenerator(rescale=1./255)
		gen = ig.ImageGenerator(self.all_paths, self.all_labels, datagen, self.n_classes,
                 target_size=target_size , color_mode='rgb',
                 batch_size=batch_size, shuffle=shuffle, seed=random_state)
		return gen


	def get_train_test_generators(self, train_size=.8, batch_size=16, grayscale=False, target_size=(150,150), random_state=42):
		path_train, path_test, label_train, label_test = partition_train_test(
															self.all_paths, self.all_labels, train_size, random_state)

		# TODO: make this variable so not forced to augment data on creation
		train_datagen = ImageDataGenerator(rescale=1./255,
									        shear_range=0.2,
									        zoom_range=0.2,
									        horizontal_flip=True)

		test_datagen = ImageDataGenerator(rescale=1./255)


		train_gen = ig.ImageGenerator(path_train, label_train, train_datagen, self.n_classes,
                 target_size=target_size , color_mode='rgb',
                 batch_size=batch_size, shuffle=False, seed=random_state)

		test_gen = ig.ImageGenerator(path_test, label_test, test_datagen, self.n_classes,
                 target_size=target_size, color_mode='rgb',
                 batch_size=batch_size, shuffle=False, seed=random_state)

		return train_gen, test_gen


############################# STATIC METHODS ################################
def partition_train_test(X, Y, train_size=.8, random_state=42):
	'''
	returns X_train, X_test, y_train, y_test
	'''
	print 'partitioning data with train size', int(len(X)*train_size), 'and test size', int(len(X)*(1 - train_size))
	return train_test_split(X, Y, train_size=train_size, random_state=random_state)



