import numpy as np
import keras
import tensorflow as tf
from keras import backend
from models.cifarmodel import cifar_model
from models.mnistmodel import mnist_model
from loaddata import load_cifar, load_mnist
from models.vae import vae_model_mnist, vae_model_cifar
from art.attacks.evasion import FastGradientMethod
from art.classifiers import KerasClassifier
import matplotlib.pyplot as plt
import argparse
import logging

parser = argparse.ArgumentParser(description='run adversarial defense using VAE')
parser.add_argument('--dataset', type=str, default='cifar', help='dataset type: cifar or mnist')
args = parser.parse_args()

# Configure a logger to capture outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if args.dataset == 'cifar':
    # Load CIFAR dataset
    train_x, train_y, test_x, test_y = load_cifar()
elif args.dataset == 'mnist':
    train_x, train_y, test_x, test_y = load_mnist()
else:
    raise ValueError('Dataset type has to be `cifar` or `mnist`')

if args.dataset == 'cifar':
    logger.info('CIFAR-10 Dataset')
    # cifar model
    cifar_model = cifar_model()
    optimizer = keras.optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=False)
    cifar_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    cifar_model.load_weights("trained_model/cifar_model.h5")
    classifier = KerasClassifier(model=cifar_model)

    # Load VAE
    VAE_model = vae_model_cifar()
    VAE_model.compile(optimizer='adam')
    VAE_model.load_weights("vae_cifar.h5")
else:
    logger.info('MNIST Dataset')
    # Mnist model
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    mnist_model, logits = mnist_model(input_ph=x, logits=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    mnist_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    mnist_model.load_weights("trained_model/mnist_model.h5")
    classifier = KerasClassifier(model=mnist_model)

    # Load VAE
    VAE_model = vae_model_mnist()
    VAE_model.compile(optimizer='adam')
    VAE_model.load_weights("vae_mnist.h5")

test_x = test_x[1000:]
test_y = test_y[1000:]
for p in range(0,31,5):
    X_adv = test_x
    if p > 0:
        # Generate adversarial images
        attack = FastGradientMethod(classifier, eps=float(p*0.01))
        X_adv = attack.generate(test_x)

    # Accuracy and fooling rate
    preds_X_adv = np.argmax(classifier.predict(X_adv), axis=1)
    acc_X_adv = np.sum(preds_X_adv == np.argmax(test_y, axis=1)) / test_y.shape[0]

    X_vae = VAE_model.predict(X_adv, batch_size=500)

    preds_X_vae = np.argmax(classifier.predict(X_vae), axis=1)
    acc_X_vae = np.sum(preds_X_vae == np.argmax(test_y, axis=1)) / test_y.shape[0]
    print(float(p*0.01), acc_X_adv, acc_X_vae)
