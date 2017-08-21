from datatypes import ImageTraining as it
import numpy as np
from misc import utils
import models.pretrained as pt
import cPickle


def get_label_dict():
	clear_path = 'data/goodbadcombined/clear_leaves'
	blurry_path = 'data/goodbadcombined/blurry/'
	busy_path = 'data/goodbadcombined/busy/'
	bad_path = 'data/goodbadcombined/bad_image'
	return {0: [blurry_path, busy_path, bad_path], 1: [clear_path]}


def quality_small_model():
	label_dict = get_label_dict()
	batch_size = 16
	target_size = (256, 256)
	input_shape = target_size + (3,)

	model = CNN_models.get_multiclass_CNN(input_shape=input_shape)

	itrain = it.ImageTraining(label_dict)
	traingen, testgen = itrain.get_train_test_generators(batch_size=batch_size, target_size=target_size)

	history = model.fit_generator(
	        traingen,
	        steps_per_epoch=traingen.total_batches*2,
	        epochs=25,
	        validation_data=testgen,
	        validation_steps=testgen.total_batches)

	weights_path = 'serialized_objects/0818/small_cnn_quality_WEIGHTS_0818.h5'
	json_path = 'serialized_objects/0818/small_cnn_quality_MODEL_0818.'
	model.save_weights(weights_path)
	model_json = model.to_json()

	with open(json_path, 'w') as outfile:
		outfile.write(model_json)

	return True

def quality_large_model():
	label_dict = get_label_dict()
	batch_size = 16
	target_size = (256, 256)
	input_shape = target_size + (3,)

	itrain = it.ImageTraining(label_dict)
	traingen, testgen = itrain.get_train_test_generators(batch_size=batch_size, target_size=target_size)

	output_train = 'serialized_objects/0818/outputtrain_QUALITY.npy'
	output_val = 'serialized_objects/0818/outputval_QUALITY.npy'
	labels = 'serialized_objects/0818/labels_QUALITY.p'

	print 'CREATING OUTPUT VALUES FROM TOP LEVEL MODEL'
	pt.get_output_values(traingen, testgen, output_train, output_val, labels)

	full_model_json = 'serialized_objects/0818/model_transf_obj_QUALITY.json'
	weights_path = 'serialized_objects/0818/model_transf_weights_QUALITY.hd5'
	num_classes = len(label_dict.keys())
	epochs = 100

	print 'CREATING PREPARING TO TRAIN LOWER DENSE MODEL...'
	model_history = pt.train_top_model(full_model_json, weights_path, output_train,
                                   output_val, labels, epochs=epochs, num_classes=num_classes)

	print "COMPLETE."
	return True


if __name__ == '__main__':
	quality_large_model()
	quality_small_model()
