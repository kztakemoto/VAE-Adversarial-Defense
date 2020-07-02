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
from art.defences.preprocessor import JpegCompression
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='run adversarial defense using VAE')
parser.add_argument('--dataset', type=str, default='cifar', help='dataset type: cifar or mnist')
args = parser.parse_args()

if args.dataset == 'cifar':
    # Load CIFAR dataset
    train_x, train_y, test_x, test_y = load_cifar()
elif args.dataset == 'mnist':
    train_x, train_y, test_x, test_y = load_mnist()
else:
    raise ValueError('Dataset type has to be `cifar` or `mnist`')

if args.dataset == 'cifar':
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

# extract data
test_x = test_x[100:]
test_y = test_y[100:]

if args.dataset == 'cifar':
    p_range = range(0,11,2)
else:
    p_range = range(0,31,5)

for p in p_range:
    X_adv = test_x
    eps = float(p*0.01)
    if p > 0:
        # Generate adversarial images
        attack = FastGradientMethod(classifier, eps=eps)
        X_adv = np.clip(attack.generate(test_x), 0, 1)

    # no defense
    preds_X_adv = np.argmax(classifier.predict(X_adv), axis=1)
    acc_X_adv = np.sum(preds_X_adv == np.argmax(test_y, axis=1)) / test_y.shape[0]

    # defense using JpegCompression
    preproc = JpegCompression(clip_values=(0, 1))
    X_def, _ = preproc(X_adv)
    preds_X_def = np.argmax(classifier.predict(X_def), axis=1)
    acc_X_def = np.sum(preds_X_def == np.argmax(test_y, axis=1)) / test_y.shape[0]

    # defense using VAE
    X_vae = VAE_model.predict(X_adv, batch_size=500)
    preds_X_vae = np.argmax(classifier.predict(X_vae), axis=1)
    acc_X_vae = np.sum(preds_X_vae == np.argmax(test_y, axis=1)) / test_y.shape[0]

    print(eps, acc_X_adv, acc_X_def, acc_X_vae)
