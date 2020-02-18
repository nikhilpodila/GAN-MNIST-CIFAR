# Generative Adversarial Networks - MNIST, FMNIST and CIFAR-10 datasets
### Contributors: Nikhil Podila, Shantanil Bagchi, Mehdi A
### Mini-Project 4 - COMP 551 Applied Machine Learning - McGill University

## Abstract
In the present study, the objective is to reproduce the original results for Generative Adversarial Nets (GANs) as well as to try to improve them. In addition to the original GAN, we consider other variants such as Deep Convolutional GANs (DCGAN) and Wasserstein GANs (WGAN). The algorithms are applied to some benchmark data sets i.e. MNIST, FMNIST and CIFAR10. The reproduced results are evaluated by quantitative measure, the Parzen window-based log-likelihood estimation as described in the original paper. Our results show that the proposed algorithms outperform the results of the original paper in terms of the quantitative measure and visual appearance of the generated images. 

## Repository Structure
The repository contains 5 files:
* 3 Jupyter notebook files -
  * GAN_MNIST_Latent_Dim and Image Quantity Test + FMNIST + DCGAN for MNIST.ipynb
  * DCGAN CIFAR-10 + Parzen Window.ipynb
  * WGAN.ipynb
* 1 ReadMe file - ReadMe.md
* 1 Project writeup - writeup.pdf

## Code Usage - (Google Colab - Python 3.7)
1. Open Jupyter notebook in Google Colab using a Google account (https://colab.research.google.com)
2. Upload the relevant notebooks
3. Switch Runtime type to GPU to ensure faster execution
4. The second code block contains Google Drive connection. Ensure that connection is established and Change directory into directory where the dataset files are stored.
5. Run all the cells. 

## Alternative installations
This project contains code written using the Keras Framework on TensorFlow. On Google Colab, all the required libraries are installed by default.
In case an alternate environment is used for executing the above code, ensure that the second code block (for connecting to Google Drive) is disabled and the following libraries are installed:

* keras
* tensorflow
* numpy
* cv2 (OpenCV)
* matplotlib
* h5py
* pandas
* seaborn
* pickle
* datetime
* warnings
