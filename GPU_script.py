from models import Models, CNN_models
from DataProcessing import ImageGenerator
import sys

def run_cnn_model(store_path, epochs=50, steps_per_epoch=2000, validation_steps=800):
	train_generator, validation_generator = ImageGenerator.get_generator()

	print 'getting CNN...'
	model_cnn = CNN_models.get_cnn()
	print 'fitting model on training data'

	history = model_cnn.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

	model_cnn.save_weights(store_path) 
	return None


def Main(args):
	store_path = args[1]
	epochs = int(args[2])
	steps_per_epoch = int(args[3])
	validation_steps = int(args[4])

	run_cnn_model(store_path, epochs, steps_per_epoch, validation_steps)

if __name__ == '__main__':
	Main(sys.argv)
