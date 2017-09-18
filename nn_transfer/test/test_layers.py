import unittest

import numpy as np
import torch.nn as nn

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, PReLU, ELU
from keras.layers import Conv2DTranspose, Conv2D, Conv3D

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


class Conv2DNet(nn.Module):
    def __init__(self):
        super(Conv2DNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, 7)

    def forward(self, x):
        return self.conv(x)


class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()
        self.conv = nn.Conv3d(3, 8, 5)

    def forward(self, x):
        return self.conv(x)


class TestLayers(TransferTestCase, unittest.TestCase):

    def setUp(self):
        self.test_data = np.random.rand(2, 3, 32, 32)
        self.test_data_3d = np.random.rand(2, 3, 8, 8, 8)

    def test_batch_normalization(self):
        keras_model = Sequential()
        keras_model.add(
            BatchNormalization(input_shape=(3, 32, 32), axis=1, name='bn'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = BatchNet()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(
            keras_model, pytorch_model, self.test_data, 1e-3)

    def test_transposed_conv(self):
        keras_model = Sequential()
        keras_model.add(Conv2DTranspose(32, (2, 2), strides=(
            2, 2), input_shape=(3, 32, 32), name='trans'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = TransposeNet()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(keras_model, pytorch_model, self.test_data)

    # Tests special activation function
    def test_elu(self):
        keras_model = Sequential()
        keras_model.add(ELU(input_shape=(3, 32, 32), name='elu'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = ELUNet()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(keras_model, pytorch_model, self.test_data)

    # Tests activation function with learned parameters
    def test_prelu(self):
        keras_model = Sequential()
        keras_model.add(PReLU(input_shape=(3, 32, 32), shared_axes=(2, 3),
                              name='prelu'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = PReLUNet()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(keras_model, pytorch_model, self.test_data)

    def test_conv2d(self):
        keras_model = Sequential()
        keras_model.add(Conv2D(16, (7, 7), input_shape=(3, 32, 32),
                               name='conv'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = Conv2DNet()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(keras_model, pytorch_model, self.test_data)

    def test_conv3d(self):
        keras_model = Sequential()
        keras_model.add(Conv3D(8, (5, 5, 5), input_shape=(3, 8, 8, 8),
                               name='conv'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = Conv3DNet()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(keras_model,
                                   pytorch_model,
                                   self.test_data_3d)

    def test_keras_model_changed_as_expected(self):
        keras_model = Sequential()
        keras_model.add(Conv2D(16, (7, 7), input_shape=(3, 32, 32),
                               name='conv'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = Conv2DNet()

        weights_before = keras_model.layers[0].get_weights()[0]
        prediction_before = keras_model.predict(self.test_data)

        self.transfer(keras_model, pytorch_model)

        weights_after = keras_model.layers[0].get_weights()[0]
        prediction_after = keras_model.predict(self.test_data)

        if self.is_keras_to_pytorch():  # Keras model should be unchanged
            self.assertTrue((weights_before == weights_after).all())
            self.assertEqual(
                prediction_before.tobytes(),
                prediction_after.tobytes(),
                msg="Predictions not are exactly the same")
        else:
            self.assertFalse((weights_before == weights_after).all())
            self.assertNotEqual(
                prediction_before.tobytes(),
                prediction_after.tobytes(),
                msg="Predictions are exactly the same")


if __name__ == '__main__':
    unittest.main()
