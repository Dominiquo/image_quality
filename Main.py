from datatypes import ImageQuality
from models import Models, CNN_models
from DataProcessing import transformations, ImageGenerator
from misc import utils
import sys

def run_canny_linear_model(create_data=False):
	obj_path = 'serialized_objects/iqual_obj.p'
	pos_path =  'data/goodbadcombined/clear_leaves/'
	neg_path = 'data/goodbadcombined/bad_image/'

	# TRANSFORMATION
	if create_data:
		canny_edge = transformations.canny_edge_avg_pixels
		iqual = ImageQuality.ImageQuality(pos_path, neg_path)
		iqual.get_image_vectors(image_setting=0)
		iqual.apply_transformation(canny_edge)
		utils.store_object(iqual, obj_path)
	else:
		iqual = utils.load_object(obj_path)


	print 'getting linear model...'
	lin_model = Models.get_linearreg_model()
	print 'fitting model on training data'
	print iqual.apply_sklearn_model(model=lin_model)
	return lin_model



def run_cnn_model(store_path, epochs=50, steps_per_epoch=2000, validation_steps=800):
	train_generator, validation_generator = ImageGenerator.get_generator()

	print 'getting CNN...'
	model_cnn = CNN_models.get_binary_classification_CNN()
	print 'fitting model on training data'

	history = model_cnn.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

	model.save_weights(store_path) 
	return None


def Main(args):
	store_path = args[1]
	epochs = int(args[2])
	steps_per_epoch = int(args[3])
	validation_steps = int(args[4])

	run_cnn_model(store_path, epochs, steps_per_epoch, validation_steps)

if __name__ == '__main__':
	Main(sys.argv)
