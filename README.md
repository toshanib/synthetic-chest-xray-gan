# Synthetic Chest X-ray Generation using Vanilla GAN

## Overview
This project implements a Vanilla Generative Adversarial Network (GAN) to generate
privacy-preserving synthetic chest X-ray images. The system is designed for
educational, research, and data-augmentation purposes where real medical data
cannot be freely shared due to privacy constraints.

The project follows a complete end-to-end machine learning pipeline including:
- Data preprocessing
- Model design
- Training
- Evaluation
- Deployment using Streamlit

---

## Dataset
- Dataset: ChestMNIST (MedMNIST benchmark)
- Image Type: Chest X-ray
- Resolution: 28 × 28
- Channels: Grayscale (1 channel)

The dataset is already anonymized and suitable for academic use.

---

## Model Architecture

### Generator
- Input: 100-dimensional latent noise vector
- Fully Connected layer → feature map reshaping
- Transposed Convolution layers for upsampling
- Output activation: Tanh
- Output: Synthetic chest X-ray image (1 × 28 × 28)

### Discriminator
- Input: Chest X-ray image (real or synthetic)
- Convolutional layers for downsampling
- LeakyReLU activations
- Fully connected output with Sigmoid
- Output: Probability of image being real

---

## Training Configuration
- Loss Function: Binary Cross Entropy (BCE)
- Optimizer: Adam (lr = 0.0002, beta1 = 0.5)
- Batch Size: 64
- Epochs: 30–100 (early stopping based on visual convergence)

---

## Project Structure

```text
synthetic-chest-xray-gan/
│
├── src/
│   ├── data_loader.py
│   ├── generator.py
│   ├── discriminator.py
│   ├── train.py
│   ├── inference.py
│   ├── evaluation.py
│   └── app.py
│
├── checkpoints/
│   └── G_final.pth
│
├── samples/
│   └── inference/
│
├── reports/
│   └── Project_Report.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```


## Project Results

![photo_6311794977003999075_w](https://github.com/user-attachments/assets/6d65653c-c89e-4818-ba38-7036c7c0ceab)
![photo_6311794977003999076_x](https://github.com/user-attachments/assets/66081421-38d7-4be9-a4a8-3c82adeea4d0)


