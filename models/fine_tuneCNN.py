import numpy as np
import cPickle
from DataProcessing import ImageGenerator as imGen
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
import models.vgg_16_keras as vgg



# TODO Make this more generalized: Appears to need more computing power than I have right now
def fine_tune_CNN(store_path='serialized_objects/fine_tune_cnn_weights.h5'):
	top_model_weights_path = 'serialized_objects/bottleneck_fc_model.h5'
	img_width, img_height = 150, 150
	input_shape = (img_width, img_height, 3)
	epochs = 50
	batch_size = 16

	print 'loading base vgg16 model...'
	base_model = vgg.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
	print 'loading complete.'

	train_generator, validation_generator = imGen.get_generator()
	nb_train_samples = train_generator.samples
	nb_validation_samples = validation_generator.samples

	top_model = Sequential()
	top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(1, activation='sigmoid'))
	top_model.load_weights(top_model_weights_path)

	model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
	for layer in model.layers[:15]:
		layer.trainable = False

	model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

	model.summary()

	history = model.fit_generator(
		train_generator,
		steps_per_epoch=nb_train_samples // batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples // batch_size,
		verbose=2)	

	print 'storing weights at', store_path
	model.save_weights(store_path)
	return history

