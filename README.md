# nn-transfer

[![Build Status](https://travis-ci.org/gzuidhof/nn-transfer.svg?branch=master)](https://travis-ci.org/gzuidhof/nn-transfer)

This repository contains utilities for **converting PyTorch models to Keras**. More specifically, it allows you to copy the weights from a PyTorch model to an identical model in Keras and vice-versa.

From Keras you can then run it on the **TensorFlow**, **Theano** and **CNTK** backend. You can also convert it to a pure TensorFlow model (see [[1]](https://github.com/amir-abdi/keras_to_tensorflow) and [[2]](https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html)), allow you to choose more robust deployment options in the cloud, or even mobile devices. From Keras you can also do inference in browsers with [keras-js](https://github.com/transcranial/keras-js).

## Installation
Clone this repository, and simply run

```
pip install .
```

You need to have PyTorch and torchvision installed beforehand, see the [PyTorch website](https://www.pytorch.org) for how to easily install that.

## Tests

To run the unit and integration tests:

```
python setup.py test
# OR, if you have nose2 installed,
nose2
```

There is also Travis CI which will automatically build every commit, see the button at the top of the readme. You can test the direction of weight transfer individually using the `TEST_TRANSFER_DIRECTION` environment variable, see `.travis.yml`.

## How to use

See [example.ipynb](example.ipynb) for a small tutorial on how to use this library.

## Code guidelines

* This repository is fully PEP8 compliant, I recommend `flake8`.
* It works for both Python 2 and 3.
