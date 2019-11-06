#! /usr/bin/env python
"""
Thai word Segmentation using Convolutional Neural Network
"""

from setuptools import setup

setup(
    name='deepcut',
    packages=['deepcut'],
    include_package_data=True,
    version='0.7.0.0',
    install_requires=['tensorflow>=2.0.0', 'pandas',
                      'scipy', 'numpy', 'scikit-learn'],
    license='MIT',
    description='A Thai word tokenization library using Deep Neural Network',
    author='Rakpong Kittinaradorn',
    author_email='r.kittinaradorn@gmail.com',
    url='https://github.com/rkcosmos/deepcut',
    download_url='https://github.com/rkcosmos/deepcut.git',
    keywords=['thai word segmentation deep learning neural network development'],
    classifiers=[
        'Development Status :: 3 - Alpha'
    ],
)
