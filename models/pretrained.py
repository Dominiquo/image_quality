import numpy as np
import cPickle
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
import models.vgg_16_keras as vgg

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'serialized_objects/bottleneck_fc_model.h5'
bottleneck_trian_npy = 'serialized_objects/bottleneck_features_train.npy'
bottleneck_val_npy = 'serialized_objects/bottleneck_features_validation.npy'
train_val_labels_p = 'serialized_objects/top_model_sizes.p'

train_data_dir = 'data/goodbadcombined/train/'
validation_data_dir = 'data/goodbadcombined/validation/'
epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    print 'loading vgg model...'
    model = vgg.VGG16(include_top=False, weights='imagenet')

    print 'creating generator from:', train_data_dir
    print 'image target size', (img_width, img_height), 'and batch size:', batch_size
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    train_steps = generator.samples / batch_size
    train_samples = train_steps*batch_size
    train_labels = generator.classes[:train_samples]

    print 'predict_generator bottleneck features train...'
    bottleneck_features_train = model.predict_generator(
        generator, train_steps)

    np.save(open(bottleneck_trian_npy, 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    validation_steps = generator.samples / batch_size
    validation_samples = validation_steps * batch_size
    validation_labels = generator.classes[:validation_samples]

    print 'predict_generator bottleneck features validation...'
    bottleneck_features_validation = model.predict_generator(
        generator, validation_steps)
    np.save(open(bottleneck_val_npy, 'w'),
            bottleneck_features_validation)

    cPickle.dump((train_labels, validation_labels), open(train_val_labels_p, 'wb'))


def train_top_model():
    print 'train top model'
    train_labels, validation_labels = cPickle.load(open(train_val_labels_p, 'rb'))
    print 'loaded', len(train_labels), 'training labels and', len(validation_labels), 'validation labels.'
    print 'loading bottleneck_trian_npy...'
    train_data = np.load(open(bottleneck_trian_npy))
    print 'loaded', len(train_data), 'training data samples'
    print 'loading bottleneck_val_npy...'
    validation_data = np.load(open(bottleneck_val_npy))
    print 'loaded', len(validation_data), 'validation data samples'

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    print 'fitting model...'
    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    print 'storing weights...'
    model.save_weights(top_model_weights_path)
    return history