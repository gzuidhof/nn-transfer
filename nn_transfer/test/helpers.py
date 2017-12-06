from __future__ import print_function
import os

import numpy as np
import torch
from torch.autograd import Variable

from .. import transfer

if 'TEST_TRANSFER_DIRECTION' in os.environ:
    TRANSFER_DIRECTION = os.environ['TEST_TRANSFER_DIRECTION'].lower()
else:
    TRANSFER_DIRECTION = 'keras2pytorch'

print(TRANSFER_DIRECTION, "tests")


def set_seeds():
    torch.manual_seed(0)
    np.random.seed(0)


class TransferTestCase(object):
    def assertEqualPrediction(
            self, keras_model, pytorch_model, test_data, delta=1e-6):

        # Make sure the pytorch model is in evaluation mode (i.e. no dropout)
        pytorch_model.eval()

        test_data = test_data.astype(np.float32, copy=False)
        test_data_tensor = Variable(
            torch.from_numpy(test_data),
            requires_grad=False)

        keras_prediction = keras_model.predict(test_data)
        pytorch_prediction = pytorch_model(test_data_tensor).data.numpy()

        self.assertEqual(keras_prediction.shape, pytorch_prediction.shape)
        for v1, v2 in zip(keras_prediction.flatten(),
                          pytorch_prediction.flatten()):
            self.assertAlmostEqual(v1, v2, delta=delta)
        return keras_prediction, pytorch_prediction

    def is_keras_to_pytorch(self):
        return TRANSFER_DIRECTION == 'keras2pytorch'

    def transfer(self, keras_model, pytorch_model, verbose=False):

        if self.is_keras_to_pytorch():
            transfer.keras_to_pytorch(keras_model,
                                      pytorch_model,
                                      verbose=verbose)
        else:
            transfer.pytorch_to_keras(pytorch_model,
                                      keras_model,
                                      verbose=verbose)
