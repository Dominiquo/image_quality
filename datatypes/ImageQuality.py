import cv2
import numpy as np
import os
import cPickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from misc import utils


class ImageQuality(object):
	def __init__(self, positive_images_dir, negaitve_images_dir):
		self.pos_dir = positive_images_dir
		self.neg_dir = negaitve_images_dir
		self.X = None
		self.y = None
		self.model = None

	def get_image_vectors(self, image_setting=1):
		print 'loading positive data...'
		pos_data = np.array([d for d in utils.load_data(self.pos_dir, image_setting)])
		pos_labels = np.array([1 for v in range(len(pos_data))])
		print 'positive data shape:', np.shape(pos_data)
		print 'positive labels shape:', np.shape(pos_labels)
		print 'loading negative data...'
		neg_data = np.array([d for d in utils.load_data(self.neg_dir)])
		neg_labels = np.array([0 for v in range(len(neg_data))])
		print 'negative data shape:', np.shape(neg_data)
		print 'negative labels shape:', np.shape(neg_labels)
		self.X = np.concatenate((pos_data, neg_data), axis=0)
		self.y = np.ravel(np.concatenate((pos_labels, neg_labels), axis=0))
		return self.X, self.y


	def apply_transformation(self, matrix_transform):
		print 'applying transformation to original feature matrix of shape', self.X.shape
		self.X = matrix_transform(self.X)
		print 'new shape of feature matrix:', self.X.shape
		return self.X


	def partition_train_test(self, test_size=.3, random_state=42):
		'''
		returns X_train, X_test, y_train, y_test
		'''
		print 'partitioning data with test size', int(len(self.X)*test_size), 'and train size', int(len(self.X)*(1 - test_size))
		try:
			return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
		except Exception as e:
			print e
			return None
			

	def apply_sklearn_model(self, model=None, test_size=.3, random_state=42):
		if model != None:
			self.model = model
		X_train, X_test, y_train, y_test = self.partition_train_test(test_size=test_size, random_state=random_state)
		self.model.fit(X_train, y_train)
		return self.model.score(X_test, y_test)



############## CALLABLE FUNCTION WHEN RUNNING ###############


def cross_validation_scoring(estimator, X, y, threshold=.5):
	'''
	positive recall
	negative recall
	positve precision
	negative precision
	'''
	y_proba = estimator.predict_proba(X)
	total_vals = len(X)
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	for i, (neg, pos) in enumerate(y_proba):
		actual_val = y[i]
		if pos >= threshold and actual_val == 1:
			tp += 1
		elif pos < threshold and actual_val == 0:
			tn += 1
		elif pos < threshold and actual_val == 1:
			fn += 1
		elif pos >= threshold and actual_val == 0:
			fp += 1

	pos_recall = (float(tp)/(tp + fn))
	try:
		neg_recall = (float(tn)/(tn + fp))
	except Exception as e:
		neg_recall = -1
	pos_prec = (float(tp)/(tp + fp))
	try:
		neg_prec = (float(tn)/(tn + fn))
	except Exception as e:
		neg_prec = -1

	cross_vals = [(pos_recall, neg_recall, pos_prec, neg_prec)]
	cross_val_scoring_print(cross_vals)
	return pos_prec


def cross_val_scoring_print(score_array):
	for pos_recall, neg_recall, pos_prec, neg_prec in score_array:
		print '____________________________________'
		print 'positive recall:', pos_recall
		print 'positive precision:', pos_prec
		print 'negative recall:', neg_recall
		print 'negative precision:', neg_prec
	return None



