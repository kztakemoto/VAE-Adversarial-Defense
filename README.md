# VAE-Adversarial-Defense

This repository is a forked version of [VAE-Adversarial-Defense](https://github.com/Roy-YL/VAE-Adversarial-Defense).

### Requirements

- Python 3.7
- Tensorflow and Keras
- IBM Adversarial Robustness Toolbox 1.1.0
- Sklearn
- Scipy
- matplotlib

## MNIST and CIFAR-10
### Install IBM Adversarial Robustness Toolbox 1.1.0
```
pip install git+https://github.com/kztakemoto/adversarial-robustness-toolbox
```

### Train the classifiers for MNIST and CIFAR-10
```
python train_classifier.py
```

### Train the VAEs for MNIST and CIFAR-10
```
python train_vae.py
```

### To evaluate the attacks using FGSM and defenses using VAE
for MNIST
```
python evaluate_VAE_defense.py --dataset mnist
```
for CIFAR-10
```
python evaluate_VAE_defense.py --dataset cifar
```

## NIPS 2017 Defense Against Adversarial Attacks Dataset

Download the 1000 image dataset and pretrained Inception-V3 model checkpoint from the [Kaggle competition](https://www.kaggle.com/c/nips-2017-defense-against-adversarial-attack/data).

Store the images in a directory named `images`, the Inception-V3 model checkpoint in a directory named `inception-v3`.

To train the VAE models on the images, run

```shell
python train_vae.py
```

To perform FGSM and I_FGSM attacks on the images, run

```shell
python attack.py
```

The attacked images will be stored in directories with names such as `fgsm_images_0.005` where `0.005` indicates the attack hyperparameter `epsilon`.

To evaluate the defense on the attacked images, run

```shell
python evaluate.py
```

The results will be saved into a `csv` file.