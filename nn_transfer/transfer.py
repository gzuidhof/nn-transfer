from __future__ import print_function
from collections import OrderedDict

import numpy as np
import h5py
import torch
import keras

from . import util

VAR_AFFIX = ':0' if keras.backend.backend() == 'tensorflow' else ''

KERAS_GAMMA_KEY = 'gamma' + VAR_AFFIX
KERAS_KERNEL_KEY = 'kernel' + VAR_AFFIX
KERAS_ALPHA_KEY = 'alpha' + VAR_AFFIX
KERAS_BIAS_KEY = 'bias' + VAR_AFFIX
KERAS_BETA_KEY = 'beta' + VAR_AFFIX
KERAS_MOVING_MEAN_KEY = 'moving_mean' + VAR_AFFIX
KERAS_MOVING_VARIANCE_KEY = 'moving_variance' + VAR_AFFIX


def check_for_missing_layers(keras_names, pytorch_layer_names, verbose):

    if verbose:
        print("Layer names in PyTorch state_dict", pytorch_layer_names)
        print("Layer names in Keras HDF5", keras_names)

    if not all(x in keras_names for x in pytorch_layer_names):
        missing_layers = list(set(pytorch_layer_names) - set(keras_names))
        raise Exception("Missing layer(s) in Keras HDF5 that are present" +
                        " in state_dict: {}".format(missing_layers))


def keras_to_pytorch(keras_model, pytorch_model,
                     flip_filters=None, verbose=True):

    # If not specifically set, determine whether to flip filters automatically
    # for the right backend.
    if flip_filters is None:
        flip_filters = not keras.backend.backend() == 'tensorflow'

    keras_model.save('temp.h5')
    input_state_dict = pytorch_model.state_dict()
    pytorch_layer_names = util.state_dict_layer_names(input_state_dict)

    with h5py.File('temp.h5', 'r') as f:
        model_weights = f['model_weights']
        layer_names = list(map(str, model_weights.keys()))
        check_for_missing_layers(layer_names, pytorch_layer_names, verbose)
        state_dict = OrderedDict()

        for layer in pytorch_layer_names:

            params = util.dig_to_params(model_weights[layer])

            weight_key = layer + '.weight'
            bias_key = layer + '.bias'
            running_mean_key = layer + '.running_mean'
            running_var_key = layer + '.running_var'

            # Load weights (or other learned parameters)
            if weight_key in input_state_dict:
                if KERAS_GAMMA_KEY in params:
                    weights = params[KERAS_GAMMA_KEY][:]
                elif KERAS_KERNEL_KEY in params:
                    weights = params[KERAS_KERNEL_KEY][:]
                else:
                    weights = np.squeeze(params[KERAS_ALPHA_KEY][:])

                weights = convert_weights(weights,
                                          to_keras=True,
                                          flip_filters=flip_filters)

                state_dict[weight_key] = torch.from_numpy(weights)

            # Load bias
            if bias_key in input_state_dict:
                if running_var_key in input_state_dict:
                    bias = params[KERAS_BETA_KEY][:]
                else:
                    bias = params[KERAS_BIAS_KEY][:]
                state_dict[bias_key] = torch.from_numpy(
                    bias.transpose())

            # Load batch normalization running mean
            if running_mean_key in input_state_dict:
                running_mean = params[KERAS_MOVING_MEAN_KEY][:]
                state_dict[running_mean_key] = torch.from_numpy(
                    running_mean.transpose())

            # Load batch normalization running variance
            if running_var_key in input_state_dict:
                running_var = params[KERAS_MOVING_VARIANCE_KEY][:]
                state_dict[running_var_key] = torch.from_numpy(
                    running_var.transpose())

    pytorch_model.load_state_dict(state_dict)


def pytorch_to_keras(pytorch_model, keras_model,
                     flip_filters=False, flip_channels=None, verbose=True):

    if flip_channels is None:
        flip_channels = not keras.backend.backend() == 'tensorflow'

    keras_model.save('temp.h5')
    input_state_dict = pytorch_model.state_dict()
    pytorch_layer_names = util.state_dict_layer_names(input_state_dict)

    with h5py.File('temp.h5', 'a') as f:
        model_weights = f['model_weights']
        target_layer_names = list(map(str, model_weights.keys()))
        check_for_missing_layers(
            target_layer_names,
            pytorch_layer_names,
            verbose)

        for layer in pytorch_layer_names:

            params = util.dig_to_params(model_weights[layer])

            weight_key = layer + '.weight'
            bias_key = layer + '.bias'
            running_mean_key = layer + '.running_mean'
            running_var_key = layer + '.running_var'

            # Load weights (or other learned parameters)
            if weight_key in input_state_dict:
                weights = input_state_dict[weight_key].numpy()
                weights = convert_weights(weights,
                                          to_keras=False,
                                          flip_filters=flip_filters,
                                          flip_channels=flip_channels)

                if KERAS_GAMMA_KEY in params:
                    params[KERAS_GAMMA_KEY][:] = weights
                elif KERAS_KERNEL_KEY in params:
                    params[KERAS_KERNEL_KEY][:] = weights
                else:
                    weights = weights.reshape(params[KERAS_ALPHA_KEY][:].shape)
                    params[KERAS_ALPHA_KEY][:] = weights

            # Load bias
            if bias_key in input_state_dict:
                bias = input_state_dict[bias_key].numpy()
                if running_var_key in input_state_dict:
                    params[KERAS_BETA_KEY][:] = bias
                else:
                    params[KERAS_BIAS_KEY][:] = bias

            # Load batch normalization running mean
            if running_mean_key in input_state_dict:
                running_mean = input_state_dict[running_mean_key].numpy()
                params[KERAS_MOVING_MEAN_KEY][:] = running_mean

            # Load batch normalization running variance
            if running_var_key in input_state_dict:
                running_var = input_state_dict[running_var_key].numpy()
                params[KERAS_MOVING_VARIANCE_KEY][:] = running_var

    # pytorch_model.load_state_dict(state_dict)
    keras_model.load_weights('temp.h5')


def convert_weights(weights, to_keras, flip_filters, flip_channels=False):

    if len(weights.shape) == 3:  # 1D conv
        weights = weights.transpose()
        if flip_filters:
            weights = weights[..., ::-1].copy()
    if len(weights.shape) == 4:  # 2D conv
        if to_keras:  # K C F F
            weights = weights.transpose(3, 2, 0, 1)
        else:
            weights = weights.transpose(2, 3, 1, 0)

        if flip_channels:
            weights = weights[::-1, ::-1]

        if flip_filters:
            weights = weights[..., ::-1, ::-1].copy()
    elif len(weights.shape) == 5:  # 3D conv
        weights = weights.transpose(4, 3, 0, 1, 2)
        if flip_filters:
            weights = weights[..., ::-1, ::-1, ::-1].copy()
    else:
        weights = weights.transpose()

    return weights
