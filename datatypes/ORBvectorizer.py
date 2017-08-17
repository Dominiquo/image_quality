from misc import utils
from datetime import datetime
import cv2
import numpy as np
from numpy.linalg import norm
import cv2
import glob
import os



class ORBClustering(object):
    def __init__(self, X_train, X_test):
        self.train = X_train
        self.test = X_test

    def get_train_test_vectors(self, num_keypoints=500, feature_size=120, featuremask=None, store_path=''):
        dictionary_exists = os.path.isfile(store_path)
        if not dictionary_exists:
            print 'adding images to bag of words trainer...'
            bag_of_words = self.batch_BOW(num_keypoints, feature_size, featuremask)
            print 'clustering bag of words descriptors...'
            dictionary = np.uint8(bag_of_words.cluster())
            if store_path != '': utils.store_object(dictionary, store_path)
        else:
            print 'loading stored data from', store_path
            dictionary = utils.load_object(store_path)

        print 'getting feature extractor...'
        featureExtractor = self.batch_feature_extractor(dictionary)
        return self.apply_orb_extractor(featureExtractor, num_keypoints, feature_size, featuremask)

    def batch_BOW(self, num_keypoints, feature_size, featuremask):
        bag_of_words = cv2.BOWKMeansTrainer(feature_size)
        orb = cv2.ORB(num_keypoints)
        for img in self.train:
            try:
                keypoints, descriptors = orb.detectAndCompute(img.astype('uint8'), mask=featuremask)
                bag_of_words.add(np.float32(descriptors)) # Convert to float for KMeans
            except Exception as e:
                print 'No descriptors found for image. Possibly run without reducing the image size for results'
                print "EXCEPTION MESSAGE:", e
                pass
        return bag_of_words    

    def batch_feature_extractor(self, dictionary):
        # Define extractor and matcher
        dExtractor = cv2.DescriptorExtractor_create('ORB')
        dMatcher = cv2.DescriptorMatcher_create('BruteForce-Hamming')
        featureExtractor = cv2.BOWImgDescriptorExtractor(dExtractor, dMatcher)
        featureExtractor.setVocabulary(dictionary)
        return featureExtractor

    def apply_orb_extractor(self, featureExtractor, num_keypoints, feature_size, featuremask):
        orb = cv2.ORB(num_keypoints)
        train_points = len(self.train)
        test_points = len(self.test)
        extracted_train = np.empty((train_points, feature_size))
        extracted_test = np.empty((test_points, feature_size))

        print 'getting features for', train_points, 'training points...'
        for i in xrange(train_points):
            img = self.train[i]
            features = featureExtractor.compute(img, orb.detect(img, mask=featuremask))
            extracted_train[i] = features

        print 'getting features for', test_points, 'test points...'
        for i in xrange(test_points):
            img = self.test[i]
            features = featureExtractor.compute(img, orb.detect(img, mask=featuremask))
            extracted_test[i] = features

        return extracted_train, extracted_test


