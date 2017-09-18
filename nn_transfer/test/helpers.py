import numpy as np
import torch
from torch.autograd import Variable

from .. import transfer


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

    def transfer(self, keras_model, pytorch_model, verbose=False):
        transfer.keras_to_pytorch(keras_model,
                                  pytorch_model,
                                  verbose=verbose)
