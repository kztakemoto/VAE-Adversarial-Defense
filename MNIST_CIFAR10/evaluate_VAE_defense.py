import numpy as np
import keras
from keras import backend
from models.cifarmodel import cifar_model
from models.mnistmodel import mnist_model
from loaddata import load_cifar, load_mnist
from models.vae import vae_model_mnist, vae_model_cifar
from jpeg import jpeg
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
import matplotlib.pyplot as plt
import argparse

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
    mnist_model, logits = mnist_model(input_ph=x, logits=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    mnist_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    mnist_model.load_weights("trained_model/mnist_model.h5")

    # Load VAE
    VAE_model = vae_model_mnist()
    VAE_model.compile(optimizer='adam')
    VAE_model.load_weights("vae_mnist.h5")

# Generate adversarial images
logger.info('Craft adversarial images with FGSM')
attack = FastGradientMethod(classifier, eps=0.1)
X_adv = attack.generate(test_x)

# Accuracy and fooling rate
preds_test_x = np.argmax(classifier.predict(test_x), axis=1)
acc = np.sum(preds_test_x == np.argmax(test_y, axis=1)) / test_y.shape[0]
logger.info('Accuracy on original test images: %.2f%%', (acc * 100))

logger.info('Encodeing and decoding clean test images using VAE')
X_vae = VAE_model.predict(test_x, batch_size=500)

preds_X_vae = np.argmax(classifier.predict(X_vae), axis=1)
acc = np.sum(preds_X_vae == np.argmax(test_y, axis=1)) / test_y.shape[0]
logger.info('Accuracy on test images encoded and decoded using VAE: %.2f%%', (acc * 100))

preds_X_adv = np.argmax(classifier.predict(X_adv), axis=1)
fooling_rate = np.sum(preds_X_adv != np.argmax(test_y, axis=1)) / test_y.shape[0]
logger.info('Fooling rate of FGSM attacks on test images: %.2f%%', (fooling_rate  * 100))

logger.info('Adversarial defense using VAE')
X_adv_vae = VAE_model.predict(X_adv, batch_size=500)

preds_X_adv_vae = np.argmax(classifier.predict(X_adv_vae), axis=1)
fooling_rate = np.sum(preds_X_vae != np.argmax(test_y, axis=1)) / test_y.shape[0]
logger.info('Fooling rate on after adversarial defenses using VAE: %.2f%%', (fooling_rate  * 100))

# Plot examples
# normalization for image plot
def norm(x):
    #return (x - np.min(x)) / (np.max(x) - np.min(x))
    return np.clip(x, 0, 1)

if args.dataset == 'cifar':
    # set CIFAR-10 labels
    label = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
else
    # set MNIST labels
    label = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

logger.info('Plot the orignail image, perturbation, and adversarial image')
# select test images
idx_sample_imgs = np.array([1, 12, 123])
fig, ax = plt.subplots(len(idx_sample_imgs), 3)
for i, idx_img in enumerate(idx_sample_imgs):
    ax[i][0].imshow(norm(test_x[idx_img]))
    ax[i][0].axis('off')
    ax[i][0].set_title(label[np.argmax(test_y[idx_img])])
    ax[i][1].imshow(norm(X_adv[idx_img]))
    ax[i][1].axis('off')
    ax[i][1].set_title(label[np.argmax(preds_X_adv[idx_img])])
    ax[i][2].imshow(norm(X_vae[idx_img]))
    ax[i][2].axis('off')
    ax[i][2].set_title(label[np.argmax(preds_X_vae[idx_img])])

if args.dataset == 'cifar':
    plt.savefig('plot_cifar.png')
else
    plt.savefig('plot_mnist.png')
