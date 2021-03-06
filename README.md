# mnistgan
Creating my first GAN for MNIST data set. This program creates handwritten digits after learning from MNIST. I am using TensorFlow 2.0.0 and python 3.7.6.

## Prerequisites
[Python 3.7.6](https://www.python.org/downloads/release/python-376/)  
[TensorFlow 2.0.0](https://www.tensorflow.org/install/pip)  
[TensorBoard 2.0.0](https://stackoverflow.com/questions/33634008/how-do-i-install-tensorflows-tensorboard) (should come with tensorflow install)  
[SciPy](https://scipy.org/install.html)  

## Usage
For now, simply `git clone` the project into a directory and run `python main.py`. If you want to see the TensorBoard analytics, open another console and type `tensorboard --logdir ./logs/gradient_tape`.

## Current State
Aeshetic's wise, I just have to fix an issue with the matplotlib windows being blank, unless I want to add graphs to track the gradients which I will do later on down the road. As for the actual algorithm, I want to normalize the data from the generator into the discriminator but otherwise I'll be ready to start moving onto a bigger and better version of this.

## Method
This project uses a Deep Convolution GAN (DCGAN) with a Wasserstein loss function to learn to produce realistic handwritten digits.

## Acknowledgements
This program implements a Deep Convolutional Generative Adversarial Network (DCGAN) as laid out by [Radford et al. (2015)](https://arxiv.org/abs/1511.06434v2). The structure they suggest has been modified a bit in this project to fit with the 28x28 images provided by mnist. I have also had some help from the [TensorFlow page about DCGANs](https://www.tensorflow.org/tutorials/generative/dcgan) to help layout the steps of the algorithm (this project is also meant to help me learn tensorflow).