import torch.nn as nn
import torch.nn.functional as F

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

K.set_image_data_format('channels_first')


class SimpleNetPytorch(nn.Module):
    def __init__(self):
        super(SimpleNetPytorch, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.fc1 = nn.Linear(6 * 14 * 14, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


def simplenet_keras():
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=(1, 32, 32),
                     name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation=None, name='fc1'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD())

    return model
