import unittest

import torch
from torch.autograd import Variable

from .. import transfer

from .architectures.lenet import lenet_keras, LeNetPytorch
from .architectures.simplenet import simplenet_keras, SimpleNetPytorch
from .architectures.vggnet import vggnet_keras, vggnet_pytorch
from .architectures.unet import unet_keras, UNetPytorch


class TestArchitectures(unittest.TestCase):

    def setUp(self):
        test_data = torch.rand(4, 1, 32, 32)
        self.test_data_keras = test_data.numpy()
        self.test_data_pytorch = Variable(test_data, requires_grad=False)

        vgg_test_data = torch.rand(1, 3, 224, 224)
        self.vgg_test_data_keras = vgg_test_data.numpy()
        self.vgg_test_data_pytorch = Variable(
            vgg_test_data, requires_grad=False)

    def test_simplenet(self):
        keras_model = simplenet_keras()
        pytorch_model = SimpleNetPytorch()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)

        keras_prediction = keras_model.predict(self.test_data_keras)
        pytorch_prediction = pytorch_model(self.test_data_pytorch).data.numpy()

        self.assertEqual(keras_prediction.shape, pytorch_prediction.shape)
        for v1, v2 in zip(keras_prediction.flatten(),
                          pytorch_prediction.flatten()):
            self.assertAlmostEqual(v1, v2, delta=1e-6)

    def test_lenet(self):
        keras_model = lenet_keras()
        pytorch_model = LeNetPytorch()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)

        keras_prediction = keras_model.predict(self.test_data_keras)
        pytorch_prediction = pytorch_model(self.test_data_pytorch).data.numpy()

        self.assertEqual(keras_prediction.shape, pytorch_prediction.shape)
        for v1, v2 in zip(keras_prediction.flatten(),
                          pytorch_prediction.flatten()):
            self.assertAlmostEqual(v1, v2, delta=1e-6)

    def test_unet(self):
        keras_model = unet_keras()
        pytorch_model = UNetPytorch()
        pytorch_model.eval()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)

        keras_prediction = keras_model.predict(self.vgg_test_data_keras[:, :1])
        pytorch_prediction = pytorch_model(
            self.vgg_test_data_pytorch[:, :1]).data.numpy()

        self.assertEqual(keras_prediction.shape, pytorch_prediction.shape)
        for v1, v2 in zip(keras_prediction.flatten(),
                          pytorch_prediction.flatten()):
            self.assertAlmostEqual(v1, v2, delta=1e-6)

    def test_vggnet(self):
        return
        keras_model = vggnet_keras()
        pytorch_model = vggnet_pytorch()
        pytorch_model.eval()

        transfer.keras_to_pytorch(keras_model, pytorch_model, verbose=False)

        keras_prediction = keras_model.predict(self.vgg_test_data_keras)
        pytorch_prediction = pytorch_model(
            self.vgg_test_data_pytorch).data.numpy()

        self.assertEqual(keras_prediction.shape, pytorch_prediction.shape)
        for v1, v2 in zip(keras_prediction.flatten(),
                          pytorch_prediction.flatten()):
            self.assertAlmostEqual(v1, v2, delta=1e-6)


if __name__ == '__main__':
    unittest.main()
