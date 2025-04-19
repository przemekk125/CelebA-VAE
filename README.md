# 🎭 Variational Autoencoder for Facial Representation Learning (CelebA)

This repository contains the code and documentation for my final project in Machine Learning. The project focuses on building a **Variational Autoencoder (VAE)** to learn latent representations of facial features using the [CelebA dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

By training a VAE on this dataset, the model aims to capture complex facial attributes such as pose, expression, hairstyle, and more, enabling tasks like face generation and attribute manipulation.

---

## 📚 Project Overview

- **Goal**: Develop a VAE model that can learn meaningful latent representations from facial images.
- **Dataset**: [CelebA (CelebFaces Attributes Dataset)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - Over 200,000 celebrity images.
  - Each image annotated with 40 binary attributes (e.g., smiling, wearing glasses).
  - Images exhibit large pose variations and background clutter.
- **Why CelebA?**: Its diversity and richness make it ideal for training models that require understanding of complex facial features.

---

## 🛠️ Features

- Implementation of a VAE architecture using **TensorFlow**.
- Data preprocessing tailored for the CelebA dataset.
- Training pipeline with customizable parameters.
- Visualization tools for:
  - Reconstructed images.
  - Latent space interpolations.
  - Attribute manipulations.

---
