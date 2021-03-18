# coding: utf-8
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.backend import clear_session
import numpy as np
import os
BASE_PATH = os.path.dirname(os.path.abspath(__file__))


PATH = BASE_PATH + "/weighted_features"


def check_class(name):
    top_model_weights_path = PATH + '/bottleneck_fc_model.h5'

    train_data_dir = BASE_PATH + "/train/"

    validation_data_dir = BASE_PATH + '/validation/'

    my_path = BASE_PATH

    # batch size used by flow_from_directory and predict_generator
    batch_size = 6

    # model = applications.VGG16(include_top=False, weights='imagenet')

    img_width, img_height = 128, 128

    datagen_top = ImageDataGenerator(rescale=1.0 / 255)

    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    # nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # load the bottleneck features saved earlier

    # get the class labels for the training data, in the original order
    # train_labels = generator_top.classes

    # convert the training labels to categorical vectors
    # train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    # nb_validation_samples = len(generator_top.filenames)

    # validation_labels = generator_top.classes

    # validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    image_path = my_path + "/" + name

    # orig = cv2.imread(image_path)

    print("[INFO] loading and pre processing image...")
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    file = open('prediction.txt', 'w')
    file.write(str(bottleneck_prediction))
    file.close()
    # print(bottleneck_prediction, 'Probability')
    print()
    # print(np.argmax(bottleneck_prediction[0]), "index of max probability")
    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final classification
    class_predicted = model.predict_classes(bottleneck_prediction)
    # print(class_predicted, "hello ")
    print('maxxxxxx', np.argmax(bottleneck_prediction[0]))
    probabilities = model.predict_proba(bottleneck_prediction)
    print(probabilities)
    print('sum', sum(probabilities[0]))
    probability = (100 * (np.max(probabilities) / sum(probabilities[0])))
    inID = class_predicted[0]

    class_dictionary = generator_top.class_indices

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]
    clear_session()

    return label, probability


# print(check_class("ntc.png"))