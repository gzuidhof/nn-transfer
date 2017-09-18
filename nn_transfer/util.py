from collections import OrderedDict

_WEIGHT_KEYS = ['kernel', 'beta', 'alpha']
_WEIGHT_KEYS += [key+':0' for key in _WEIGHT_KEYS]


def state_dict_layer_names(state_dict):
    layer_names = [".".join(k.split('.')[:-1]) for k in state_dict.keys()]
    # Order preserving unique set of names
    return list(OrderedDict.fromkeys(layer_names))


def _contains_weights(keras_h5_layer):
    for key in _WEIGHT_KEYS:
        if key in keras_h5_layer:
            return True
    return False


def dig_to_params(keras_h5_layer):
    # Params are hidden many layers deep in keras HDF5 files for
    # some reason. e.g. h5['model_weights']['conv1']['dense_1'] \
    # ['dense_2']['dense_3']['conv2d_7']['dense_4']['conv1']
    while not _contains_weights(keras_h5_layer):
        keras_h5_layer = keras_h5_layer[list(keras_h5_layer.keys())[0]]

    return keras_h5_layer
