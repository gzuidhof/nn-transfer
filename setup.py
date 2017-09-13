from setuptools import setup

setup(
    name='nn_transfer',
    version='0.1.0',
    description='Transfer weights between Keras and PyTorch.',
    install_requires=[
        'numpy',
        'keras',
        'h5py',
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
    packages=['nn_transfer'],
)
