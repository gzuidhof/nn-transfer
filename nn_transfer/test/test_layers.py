import unittest

import torch
import torch.nn as nn
from torch.autograd import Variable

import keras
from keras.models import Sequential
from keras.layers import BatchNormalization, ELU

from .. import transfer

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


class TestLayers(unittest.TestCase):

    def setUp(self):
        test_data = torch.rand(1, 3, 32, 32)
        self.test_data_keras = test_data.numpy()
        self.test_data_pytorch = Variable(test_data, requires_grad=False)

    def test_batch_normalization(self):
        keras_model = Sequential()
        keras_model.add(
            BatchNormalization(
                input_shape=(
                    3,
                    32,
                    32),
                axis=1,
                name='bn'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = BatchNet()
        pytorch_model.eval()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)

        keras_prediction = keras_model.predict(self.test_data_keras)
        pytorch_prediction = pytorch_model(self.test_data_pytorch).data.numpy()

        for v1, v2 in zip(keras_prediction.flatten(),
                          pytorch_prediction.flatten()):
            self.assertAlmostEqual(v1, v2, delta=2e-3)

    # Tests activation function with learned features
    def test_elu(self):
        keras_model = Sequential()
        keras_model.add(ELU(input_shape=(3, 32, 32), name='elu'))
        keras_model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.SGD())

        pytorch_model = ELUNet()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)

        keras_prediction = keras_model.predict(self.test_data_keras)
        pytorch_prediction = pytorch_model(self.test_data_pytorch).data.numpy()

        for v1, v2 in zip(keras_prediction.flatten(),
                          pytorch_prediction.flatten()):
            self.assertAlmostEqual(v1, v2, delta=2e-3)


if __name__ == '__main__':
    unittest.main()
