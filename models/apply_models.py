import cv2
import numpy as np
from misc import utils
from models import CNN_models as cnn

def get_accuracy_threshold(model_path, data_path, pos=True, threshold=.5, size=(150,150), rescale_factor=(1./255)):
	model = cnn.get_binary_classification_CNN()
	print 'loading weights...'
	model.load_weights(model_path)
	print 'rescaling images...'
	image_data = [img*rescale_factor for img in utils.load_data(data_path)]
	print 'resizing images...'
	image_data = [cv2.resize(img, size) for img in image_data]
	print 'producing predictions...'
	return model.predict(np.array(image_data))