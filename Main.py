from datatypes import ImageQuality
from models import Models, CNN_models
from DataProcessing import transformations, ImageGenerator
from misc import utils

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



def run_cnn_model(create_data=False):
	obj_path = 'serialized_objects/iqual_obj_cnn.p'
	pos_path =  'data/goodbadcombined/clear_leaves/'
	neg_path = 'data/goodbadcombined/bad_image/'
	size = (128, 128)

	# TRANSFORMATION
	if create_data:
		resize_images = lambda img: transformations.resize_all_images(img, size)
		iqual = ImageQuality.ImageQuality(pos_path, neg_path)
		iqual.get_image_vectors()
		iqual.apply_transformation(resize_images)
		utils.store_object(iqual, obj_path)
	else:
		iqual = utils.load_object(obj_path)

	print 'getting CNN...'
	model_cnn = CNN_models.get_cnn()
	print 'fitting model on training data'
	X_train, X_test, y_train, y_test = iqual.partition_train_test(test_size=.3, random_state=42)
	model_cnn.fit(X_train, y_train)
	print model_cnn.evaluate(X_test, y_test)
	return model_cnn