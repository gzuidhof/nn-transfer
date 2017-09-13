import unittest

from .. import util


class TestUtil(unittest.TestCase):

    def test_state_dict_names(self):
        state_dict = {
            'conv1.weight': 0,
            'conv1.bias': 0,
            'fc1.weight': 0,
            'fc2.weight': 0,
            'fc2.bias': 0
        }
        layer_names = util.state_dict_layer_names(state_dict)

        self.assertListEqual(sorted(layer_names), ['conv1', 'fc1', 'fc2'])


if __name__ == '__main__':
    unittest.main()
