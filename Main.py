from misc import utils
from datatypes import ImageTraining as im


def Main():
	pos = 'data/goodbadcombined/validation/positive/'
	neg = 'data/goodbadcombined/validation/negative/'
	label_dict = {1:[pos], 0:[neg]}
	itrain = im.ImageTraining(label_dict)
	traingen, testgen = itrain.get_train_test_generators()
	return None

if __name__ == '__main__':
	Main()
