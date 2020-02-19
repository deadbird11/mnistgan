# mnistgan
Creating my first GAN for MNIST data set. This program creates handwritten digits after learning from MNIST. I am using `tensorflow` 2.0 and python 3.7.6.

# Prerequisites
[Python 3.7.6](https://www.python.org/downloads/release/python-376/)
[Tensorflow 2.1.0](https://www.tensorflow.org/install/pip)
[SciPy](https://scipy.org/install.html)

# Usage
For now, simply `git clone` the project into a directory and run `python main.py`.

# Current State
There are still some bugs so I'd wait a bit before trying it out for yourself.

# Method
This project uses a Deep Convolution GAN (DCGAN) with a Wasserstein loss function to learn to produce realistic handwritten digits.

# Acknowledgements
This program implements a Deep Convolutional Generative Adversarial Network (DCGAN) as laid out by [Radford et al. (2015)](https://arxiv.org/abs/1511.06434v2). The structure they suggest has been modified a bit in this project to fit with the 28x28 images provided by mnist. I have also had some help from the [`tensorflow` page about DCGANs](https://www.tensorflow.org/tutorials/generative/dcgan) to help layout the steps of the algorithm.