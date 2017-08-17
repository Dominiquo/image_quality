import cv2
import numpy as np
from misc import utils
from keras.preprocessing.image import ImageDataGenerator

def get_accuracy_threshold(model, data_path, size=(150,150), rescale_factor=(1./255)):
	print 'rescaling images...'
	datagen = ImageDataGenerator(rescale=rescale_factor)
	data_generator = utils.get_data_iterator(data_path, datagen, target_size=size)
	for batch, labels in data_generator:
		prediction = model.pred
	return True