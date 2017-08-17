from misc import utils
from DataProcessing import ImageGenerator as imgen
from datatypes import DataGenerator as dg
import time

def test_generator(samples=1000):
	pos = 'data/goodbadcombined/validation/positive/'
	neg = 'data/goodbadcombined/validation/negative/'
	label_dict = {1:[pos], 0:[neg]}
	x,y = utils.get_paths_labels(label_dict)
	batch_size = 16
	traingen = dg.DataGen(x, y, batch_size, True, False, (150,150))
	start = time.time()
	for i in xrange(samples):
		if i % 10 == 0:
			print i, '/', samples
		batch, labels = traingen.next()
	stop = time.time()
	print 'generated', samples, 'batches of size', batch_size, 'in', round(stop - start, 2), 'seconds'
	return None

def test_keras_generator(samples=1000):
	batch_size = 16
	traingen, valgen = imgen.get_generator(batch_size=batch_size)
	start = time.time()
	for i in xrange(samples):
		if i % 10 == 0:
			print i, '/', samples
		batch, labels = valgen.next()
	stop = time.time()
	print 'generated', samples, 'batches of size', batch_size, 'in', round(stop - start, 2), 'seconds'
	return None