from models import CNN_models
from DataProcessing import ImageGenerator
from models import fine_tuneCNN as ft_CNN
import sys

def run_basic_bin_classify_model(store_path, epochs=50, steps_per_epoch=2000, validation_steps=800):
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

	model_cnn.save_weights(store_path) 
	return None

def run_fine_tune_CNN():
	ft_CNN.fine_tune_CNN()



def Main(args):
	store_path = args[1]
	epochs = int(args[2])
	steps_per_epoch = int(args[3])
	validation_steps = int(args[4])

	run_basic_bin_classify_model(store_path, epochs, steps_per_epoch, validation_steps)

if __name__ == '__main__':
	Main(sys.argv)
