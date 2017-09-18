import unittest

import numpy as np
import torch.nn as nn

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, PReLU, ELU, Conv2DTranspose

from .. import transfer

from .helpers import TransferTestCase

keras.backend.set_image_data_format('channels_first')


class BatchNet(nn.Module):
    def __init__(self):
        super(BatchNet, self).__init__()
        self.bn = nn.BatchNorm3d(3)

    def forward(self, x):
        return self.bn(x)


class ELUNet(nn.Module):
    def __init__(self):
        super(ELUNet, self).__init__()
        self.elu = nn.ELU()

    def forward(self, x):
        return self.elu(x)


class TransposeNet(nn.Module):
    def __init__(self):
        super(TransposeNet, self).__init__()
        self.trans = nn.ConvTranspose2d(3, 32, 2, 2)

    def forward(self, x):
        return self.trans(x)


class PReLUNet(nn.Module):
    def __init__(self):
        super(PReLUNet, self).__init__()
        self.prelu = nn.PReLU(3)

    def forward(self, x):
        return self.prelu(x)


class TestLayers(TransferTestCase, unittest.TestCase):

    def setUp(self):
        self.test_data = np.random.rand(2, 3, 32, 32)

    def test_batch_normalization(self):
        keras_model = Sequential()
        keras_model.add(
            BatchNormalization(input_shape=(3, 32, 32), axis=1, name='bn'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = BatchNet()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)
        self.assertEqualPrediction(
            keras_model, pytorch_model, self.test_data, 1e-3)

    def test_transposed_conv(self):
        keras_model = Sequential()
        keras_model.add(Conv2DTranspose(32, (2, 2), strides=(
            2, 2), input_shape=(3, 32, 32), name='trans'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = TransposeNet()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)
        self.assertEqualPrediction(keras_model, pytorch_model, self.test_data)

    # Tests special activation function
    def test_elu(self):
        keras_model = Sequential()
        keras_model.add(ELU(input_shape=(3, 32, 32), name='elu'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = ELUNet()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)
        self.assertEqualPrediction(keras_model, pytorch_model, self.test_data)

    # Tests activation function with learned parameters
    def test_prelu(self):
        keras_model = Sequential()
        keras_model.add(PReLU(input_shape=(3, 32, 32), shared_axes=(2, 3),
                              name='prelu'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = PReLUNet()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)
        self.assertEqualPrediction(keras_model, pytorch_model, self.test_data)


if __name__ == '__main__':
    unittest.main()
