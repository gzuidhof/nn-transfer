from __future__ import print_function
from collections import OrderedDict

import h5py
import torch

from . import util


def keras_to_pytorch(keras_model, pytorch_model,
                     flip_filters=True, verbose=True):
    keras_model.save('temp.h5')
    target_state_dict = pytorch_model.state_dict()
    target_layer_names = util.state_dict_layer_names(target_state_dict)

    with h5py.File('temp.h5', 'r') as f:
        model_weights = f['model_weights']
        layer_names = map(str, model_weights.keys())

        if verbose:
            print("Layer names in target", target_layer_names)
            print("Layer names in Keras HDF5", layer_names)

        if not all(x in layer_names for x in target_layer_names):
            missing_layers = list(set(target_layer_names) - set(layer_names))
            raise Exception("Missing layer(s) in Keras HDF5 that are present" +
                            " in state dict: {}".format(missing_layers))

        state_dict = OrderedDict()

        for layer in target_layer_names:

            params = util.dig_to_params(model_weights[layer])

            weight_key = layer + '.weight'
            bias_key = layer + '.bias'
            running_mean_key = layer + '.running_mean'
            running_var_key = layer + '.running_var'

            if weight_key in target_state_dict:

                if running_var_key in target_state_dict:
                    weights = params['gamma'][:]
                else:
                    weights = params['kernel'][:]

                if len(weights.shape) == 4:  # Assume 2D conv
                    weights = weights.transpose(3, 2, 0, 1)
                    if flip_filters:
                        weights = weights[..., ::-1, ::-1].copy()
                elif len(weights.shape) == 5:  # Assume 3D conv
                    weights = weights.transpose(4, 3, 0, 1, 2)
                    if flip_filters:
                        weights = weights[..., ::-1, ::-1, ::-1].copy()
                else:
                    weights = weights.transpose()

                state_dict[weight_key] = torch.from_numpy(weights)

            if bias_key in target_state_dict:
                if running_var_key in target_state_dict:
                    bias = params['beta'][:]
                else:
                    bias = params['bias'][:]
                state_dict[bias_key] = torch.from_numpy(
                    bias.transpose())

            if running_mean_key in target_state_dict:
                running_mean = params['moving_mean'][:]
                state_dict[running_mean_key] = torch.from_numpy(
                    running_mean.transpose())

            if running_var_key in target_state_dict:
                running_var = params['moving_variance'][:]
                state_dict[running_var_key] = torch.from_numpy(
                    running_var.transpose())

    pytorch_model.load_state_dict(state_dict)
