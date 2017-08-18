from datatypes import ImageTraining as it
import numpy as np
from misc import utils
import models.pretrained as pt

HEALTH_IMAGES = 'healthy_images'
CGM_FILES = 'cgm_files'
CBB_FILES = 'cbb_files'
CMD_FILES = 'cmd_files'
CBSD_FILES = 'cbsd_files'


def get_disease_files():
	file_prefix = '/home/ailab/Documents/niquo/image_quality/data/Severities_Final/'

	cbb_files_L = [
	            file_prefix + 'cbb-levels/cbb_3/',
	            file_prefix + 'cbb-levels/cbb_4/',
	            file_prefix + 'cbb-levels/cbb_5/']
	cgm_files_L = [
	            file_prefix + 'cgm-levels/cgm_3/',
	            file_prefix + 'cgm-levels/cgm_4/',
	            file_prefix + 'cgm-levels/cgm_5/',]
	cmd_files_L = [
	            file_prefix + 'cmd-levels/cmd_3/',
	            file_prefix + 'cmd-levels/cmd_4/',
	            file_prefix + 'cmd-levels/cmd_5/']
	cbsd_files_L = [
	             file_prefix + 'cbsd-levels/cbsd_3/',
	             file_prefix + 'cbsd-levels/cbsd_4/',
	             file_prefix + 'cbsd-levels/cbsd_5/']
	healthy_images_L = [file_prefix + 'cbsd-levels/cbsd_1/',
	                  file_prefix + 'cmd-levels/cmd_1/',
	                  file_prefix + 'cbb-levels/cbb_1/', 
	                  file_prefix + 'healthy_1/']

	ret_dict = {HEALTH_IMAGES: healthy_images_L, CGM_FILES: cgm_files_L, CMD_FILES: cmd_files_L,
				 CBSD_FILES: cbsd_files_L, CBB_FILES: cbb_files_L}

	return ret_dict


def five_class_identification():
	image_dict = get_disease_files()
	label_dict = {0: image_dict[HEALTH_IMAGES], 1: image_dict[CBB_FILES], 2: image_dict[CMD_FILES],
					3: image_dict[CGM_FILES], 4: image_dict[CBSD_FILES]}

	for k,v in label_dict.iteritems():
		print '*******************'
		print k
		print v

	batch_size = 16 
	target_size = (300, 300)

	print 'getting trainer object...'

	im_train= it.ImageTraining(label_dict)
	traingen, testgen = im_train.get_train_test_generators(batch_size=batch_size, target_size=target_size)

	output_train = 'serialized_objects/0818/outputtrain_DIS.npy'
	output_val = 'serialized_objects/0818/outputval_DIS.npy'
	labels = 'serialized_objects/0818/labels_DIS.p'

	print 'CREATING OUTPUT VALUES FROM TOP LEVEL MODEL'
	pt.get_output_values(traingen, testgen, output_train, output_val, labels)

	full_model_json = 'serialized_objects/0818/model_object_DIS.json'
	weights_path = 'serialized_objects/0818/model_weights_DIS.hd5'
	num_classes = len(label_dict.keys())
	epochs = 100

	print 'CREATING PREPARING TO TRAIN LOWER DENSE MODEL...'
	model_history = pt.train_top_model(full_model_json, weights_path, output_train,
                                   output_val, labels, epochs=epochs, num_classes=num_classes)

	model_hist_path = 'serialized_objects/0818/model_severity_HISTORY.p'
	with open(model_hist_path, 'w') as outfile:
		cPickle.dump(model_history, outfile)

	print "COMPLETE."
	return True



if __name__ == '__main__':
	five_class_identification()