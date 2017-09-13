from torchvision.models import vgg

from keras import backend as K
from keras.models import Input, Model
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D

K.set_image_data_format('channels_first')


def vggnet_pytorch():
    return vgg.vgg16()


def vggnet_keras():

    # Block 1
    img_input = Input((3, 224, 224))
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='features.0')(img_input)
    x = Conv2D(64, (3, 3), activation='relu',
               padding='same', name='features.2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='features.5')(x)
    x = Conv2D(128, (3, 3), activation='relu',
               padding='same', name='features.7')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='features.10')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='features.12')(x)
    x = Conv2D(256, (3, 3), activation='relu',
               padding='same', name='features.14')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='features.17')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='features.19')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='features.21')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='features.24')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='features.26')(x)
    x = Conv2D(512, (3, 3), activation='relu',
               padding='same', name='features.28')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='classifier.0')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='classifier.3')(x)
    x = Dropout(0.5)(x)
    x = Dense(1000, activation=None, name='classifier.6')(x)

    # Create model.
    model = Model(img_input, x, name='vgg16')

    return model
