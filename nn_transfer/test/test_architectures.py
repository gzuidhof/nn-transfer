import unittest

import numpy as np

from .helpers import TransferTestCase, set_seeds
from .architectures.lenet import lenet_keras, LeNetPytorch
from .architectures.simplenet import simplenet_keras, SimpleNetPytorch
from .architectures.vggnet import vggnet_keras, vggnet_pytorch
from .architectures.unet import unet_keras, UNetPytorch


class TestArchitectures(TransferTestCase, unittest.TestCase):

    def setUp(self):
        self.test_data_small = np.random.rand(4, 1, 32, 32)
        self.test_data_vgg = np.random.rand(1, 3, 224, 224)
        self.test_data_unet = np.random.rand(1, 1, 224, 224)

    def test_simplenet(self):
        set_seeds()
        keras_model = simplenet_keras()
        pytorch_model = SimpleNetPytorch()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(
            keras_model,
            pytorch_model,
            self.test_data_small,
            delta=1e-3)  # These results can vary due to float imprecision

    def test_lenet(self):
        set_seeds()
        keras_model = lenet_keras()
        pytorch_model = LeNetPytorch()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(
            keras_model,
            pytorch_model,
            self.test_data_small)

    def test_unet(self):
        set_seeds()
        keras_model = unet_keras()
        pytorch_model = UNetPytorch()
        pytorch_model.eval()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(
            keras_model,
            pytorch_model,
            self.test_data_unet)

    def test_vggnet(self):
        set_seeds()
        keras_model = vggnet_keras()
        pytorch_model = vggnet_pytorch()
        pytorch_model.eval()

        self.transfer(keras_model, pytorch_model)
        self.assertEqualPrediction(
            keras_model, pytorch_model, self.test_data_vgg)


if __name__ == '__main__':
    unittest.main()
