import scipy.io as sio
import numpy as np
import keras
from models.vae import vae_model_cifar, vae_model_mnist
from loaddata import load_mnist, load_cifar
from keras import backend as K
import tensorflow as tf

# Load dataset
# MNIST
X_train, Y_train, X_test, Y_test = load_mnist()
# CIFAR
train_x, train_y, test_x, test_y = load_cifar()

# Create TF session and set as Keras backend session
sess = tf.Session()
keras.backend.set_session(sess)

vae_cifar = vae_model_cifar()
vae_mnist = vae_model_mnist()

vae_cifar.compile(optimizer='adam')
vae_mnist.compile(optimizer='adam')

vae_cifar.fit(train_x,
              epochs=200,
              batch_size=64,
              verbose=2)
vae_cifar.save_weights("vae_cifar.h5")

vae_mnist.fit(X_train,
              epochs=200,
              batch_size=64,
              verbose=2)
vae_mnist.save_weights("vae_mnist.h5")
