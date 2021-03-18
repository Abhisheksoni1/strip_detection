import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import math
import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
print(BASE_PATH)
IMG_WIDTH, IMG_HEIGHT = 128, 128

PATH = BASE_PATH + "/weighted_features"
TOP_MODELS_WEIGHT_PATH = BASE_PATH + '/weighted_features/bottleneck_fc_model.h5'
TRAIN_DATA_DIR = BASE_PATH + '/train/'
VALIDATION_DATA_DIR = BASE_PATH + '/validation/'

# define number of epochs
EPOCHS = 20

# batch size used by flow_from_directory and predict_generator
BATCH_SIZE = 6

# model generation


def deep_model():

    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    # In[30]:

    predict_size_train = int(math.ceil(nb_train_samples / BATCH_SIZE))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save(PATH + '/bottleneck_features_train.npy', bottleneck_features_train)

    # In[31]:

    generator = datagen.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(math.ceil(nb_validation_samples / BATCH_SIZE))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save(PATH + '/bottleneck_features_validation.npy', bottleneck_features_validation)

    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)

    # nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load(PATH + '/bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    # In[33]:

    generator_top = datagen_top.flow_from_directory(
        VALIDATION_DATA_DIR,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False)

    # nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load(PATH + '/bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(validation_data, validation_labels))
    print(history)
    model.save_weights(TOP_MODELS_WEIGHT_PATH)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=BATCH_SIZE, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    # print(model.summary())


deep_model()

