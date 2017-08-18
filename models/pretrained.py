# Code adapted from Keras Tutorial: https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
import numpy as np
import cPickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, model_from_json
from keras.layers import Dropout, Flatten, Dense, Activation
from datatypes import DataGenerator as dg
import models.vgg_16_keras as vgg
from keras.utils import to_categorical



def get_output_values(traingen, valgen, output_train, output_val, labels, train_batches=None, val_batches=None):
    print 'loading vgg model...'
    model = vgg.VGG16(include_top=False, weights='imagenet')
    print 'image target size', traingen.target_size, 'with batch size', traingen.batch_size
    print 'running training examples through top model...'

    if train_batches == None:
        train_batches = traingen.total_batches
        train_labels = to_categorical(traingen.classes, traingen.num_class)
        total_samples_train = len(train_labels)
    else:
        total_samples_train = get_samples_for_batches(traingen, train_batches)
        train_labels = np.empty((total_samples_train))
        for i in xrange(len(train_labels)):
            index = i % traingen.samples
            train_labels[i] = traingen.classes[index]
        train_labels = to_categorical(train_labels, traingen.num_class)

    print 'running model with', train_batches, 'training batches and', total_samples_train, 'total samples'
    print 'training labels shape:', train_labels.shape
    bottleneck_features_train = model.predict_generator(traingen, train_batches)
    print 'final shape of bottleneck_features_train:', bottleneck_features_train.shape
    np.save(open(output_train, 'w'), bottleneck_features_train)

    print 'running validation examples through top model...'
    if val_batches == None:
        val_batches = valgen.total_batches
        val_labels = to_categorical(valgen.classes, valgen.num_class)
    else: 
        total_samples_val = get_samples_for_batches(valgen, val_batches)
        val_labels = np.empty((total_samples_val))
        for i in xrange(len(val_labels)):
            index = i % total_labels
            val_labels[i] = valgen.labels[index]
        val_labels = to_categorical(val_labels, valgen.num_class)

    print 'running model with', val_batches, 'validation batches of data'
    bottleneck_features_validation = model.predict_generator(valgen, val_batches)
    print 'final shape of bottleneck_features_validation:', bottleneck_features_validation.shape
    np.save(open(output_val, 'w'), bottleneck_features_validation)

    cPickle.dump((train_labels, val_labels), open(labels, 'wb'))

    return None


def train_top_model(json_path, weights_path, output_train, output_val, labels, epochs=50, batch_size=16, num_classes=2):
    print 'train top model'
    train_labels, validation_labels = cPickle.load(open(labels, 'rb'))
    print 'loaded', len(train_labels), 'training labels and', len(validation_labels), 'validation labels.'
    print 'loading bottleneck_trian_npy...'
    train_data = np.load(open(output_train))
    print 'loaded', len(train_data), 'training data samples'
    print 'loading bottleneck_val_npy...'
    validation_data = np.load(open(output_val))
    print 'loaded', len(validation_data), 'validation data samples'

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                    optimizer='rmsprop',
                    metrics=['accuracy'])

    print 'fitting model...'
    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    print 'storing weights...'
    model.save_weights(weights_path)
    model_json = model.to_json()
    with open(json_path, 'w') as outfile:
        outfile.write(model_json)
    return history



def load_full_model(json_model_p, top_model_weights_path, input_shape):
    base_model = vgg.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    with open(json_model_p) as infile:
        print 'loading model json data from file'
        loaded_model_json = infile.read()
    print 'creating model from json object.'
    top_model = model_from_json(loaded_model_json)
    print 'loading model weights...'
    top_model.load_weights(top_model_weights_path)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    return model



def get_samples_for_batches(gen, batches):
    if batches < gen.total_batches:
        return batches*gen.batch_size
    elif batches > gen.total_batches:
        return get_samples_for_batches(gen, batches - gen.total_batches) + gen.samples
    else:
        return self.samples