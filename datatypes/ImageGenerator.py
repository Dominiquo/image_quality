from __future__ import print_function
from keras.preprocessing.image import DirectoryIterator, load_img, img_to_array
import keras.backend as K
import os
import numpy as np
from misc import utils



class ImageGenerator(DirectoryIterator):
	def __init__(self, filenames, labels, image_data_generator, num_class,
                 target_size=(256, 256), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png',
                 follow_links=False):

		if data_format is None:
		    data_format = K.image_data_format()
		self.image_data_generator = image_data_generator
		self.target_size = tuple(target_size)
		if color_mode not in {'rgb', 'grayscale'}:
		    raise ValueError('Invalid color mode:', color_mode,
		                     '; expected "rgb" or "grayscale".')
		self.color_mode = color_mode
		self.data_format = data_format
		if self.color_mode == 'rgb':
		    if self.data_format == 'channels_last':
		        self.image_shape = self.target_size + (3,)
		    else:
		        self.image_shape = (3,) + self.target_size
		else:
		    if self.data_format == 'channels_last':
		        self.image_shape = self.target_size + (1,)
		    else:
		        self.image_shape = (1,) + self.target_size
		self.classes = classes
		if class_mode not in {'categorical', 'binary', 'sparse',
		                      'input', None}:
		    raise ValueError('Invalid class_mode:', class_mode,
		                     '; expected one of "categorical", '
		                     '"binary", "sparse", "input"'
		                     ' or None.')
		self.class_mode = class_mode
		self.save_to_dir = save_to_dir
		self.save_prefix = save_prefix
		self.save_format = save_format
		self.filenames = filenames
		self.classes = labels
		self.samples = len(self.filenames)
		self.num_class = num_class
		self.total_batches = self.samples/batch_size if (self.samples % batch_size) == 0 else self.samples/batch_size + 1
		super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)


	def next(self):
	    with self.lock:
	        index_array, current_index, current_batch_size = next(self.index_generator)
	    # The transformation of images is not under thread lock
	    # so it can be done in parallel
	    batch_x = np.zeros((current_batch_size,) + self.image_shape, dtype=K.floatx())
	    grayscale = self.color_mode == 'grayscale'
	    # build batch of image data
	    for i, j in enumerate(index_array):
	        fname = self.filenames[j]
	        img = load_img(fname,
	                       grayscale=grayscale,
	                       target_size=self.target_size)
	        x = img_to_array(img, data_format=self.data_format)
	        x = self.image_data_generator.random_transform(x)
	        x = self.image_data_generator.standardize(x)
	        batch_x[i] = x
	    # optionally save augmented images to disk for debugging purposes
	    if self.save_to_dir:
	        for i in range(current_batch_size):
	            img = array_to_img(batch_x[i], self.data_format, scale=True)
	            fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
	                                                              index=current_index + i,
	                                                              hash=np.random.randint(1e4),
	                                                              format=self.save_format)
	            img.save(os.path.join(self.save_to_dir, fname))
	    # build batch of labels
	    if self.class_mode == 'input':
	        batch_y = batch_x.copy()
	    elif self.class_mode == 'sparse':
	        batch_y = self.classes[index_array]
	    elif self.class_mode == 'binary':
	        batch_y = self.classes[index_array].astype(K.floatx())
	    elif self.class_mode == 'categorical':
	        batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
	        for i, label in enumerate(self.classes[index_array]):
	            batch_y[i, label] = 1.
	    else:
	        return batch_x
	    return batch_x, batch_y

