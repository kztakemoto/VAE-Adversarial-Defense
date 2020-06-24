import numpy as np
import keras
from keras import backend
from models.cifarmodel import cifar_model
from loaddata import load_cifar
from models.vae import vae_model_mnist, vae_model_cifar
from jpeg import jpeg
from art.attacks import FastGradientMethod
from art.classifiers import KerasClassifier
import matplotlib.pyplot as plt

# Load CIFAR dataset
train_x, train_y, test_x, test_y = load_cifar()

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

logger.info('Craft adversarial images with FGSM')
attack = FastGradientMethod(classifier, eps=0.1)
X_adv = attack.generate(test_x)

# Compute accuracy
preds_test_x = np.argmax(classifier.predict(test_x), axis=1)
acc = np.sum(preds_test_x == np.argmax(test_y, axis=1)) / test_y.shape[0]
logger.info('Accuracy on test images: %.2f%%', (acc * 100))

# Compute fooling rate
preds_X_adv = np.argmax(classifier.predict(X_adv), axis=1)
fooling_rate = np.sum(preds_X_adv != np.argmax(test_y, axis=1)) / test_y.shape[0]
logger.info('Fooling rate of FGSM attacks on test images: %.2f%%', (fooling_rate  * 100))

logger.info('Adversarial defense using VAE')
X_vae = VAE_model.predict(X_adv, batch_size=500)

# Compute fooling rate
preds_X_vae = np.argmax(classifier.predict(X_vae), axis=1)
fooling_rate = np.sum(preds_X_vae != np.argmax(test_y, axis=1)) / test_y.shape[0]
logger.info('Fooling rate on after adversarial defense using VAE: %.2f%%', (fooling_rate  * 100))


# normalization for image plot
def norm(x):
    #return (x - np.min(x)) / (np.max(x) - np.min(x))
    return np.clip(x, 0, 1)

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

plt.savefig('plot_cifar.png')