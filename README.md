# Deep Learning & Machine Learning Algorithms

![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-green)](https://slab10000.github.io/deep-learning-algorithms/)

Welcome to my repository for Deep Learning and Machine Learning algorithms! 🚀

This repository serves as a personal playground and portfolio where I upload code to practice, experiment with, and master various algorithms and frameworks. It is currently **under active development**, and new models and implementations will be added regularly.

## 📂 Repository Contents

Here is an overview of what has been implemented so far:

### 🧠 Generative Adversarial Networks (GANs)
- **Conditional GANs (CGANs)**: Implementation of Conditional Generative Adversarial Networks used for generating data (e.g., MNIST images) conditioned on class labels. 
  - Location: `GANs/CGANs.ipynb`

### 📈 Classification & Regression
Try it here: https://slab10000.github.io/deep-learning-algorithms/classification-and-regression

Applied machine learning projects covering both classification and regression tasks.

- **Classification - Petfinder Dataset**: 
  - A project focusing on classifying pet adoption profiles.
  - Includes a Multi-Layer Perceptron (MLP) model saved in ONNX format (`mlp_net_model.onnx`).
  - Location: `classification-and-regression/classification/`

- **Regression - Songs Dataset**:
  - A regression analysis project, likely predicting song popularity or features.
  - Utilizes XGBoost (Extreme Gradient Boosting) and includes the serialized model (`xgboost_songs_model.onnx`).
  - Location: `classification-and-regression/regression/`

### 🎬 Neural Input Optimization (NIO) - Movie Optimization
- **Project Blockbuster: NIO Optimisation for Movies**:
  - A project that uses Neural Input Optimization (NIO) to reverse-engineer the optimal movie blueprint for maximizing both commercial success (Gross Revenue) and critical acclaim (IMDB Score).
  - **Dataset**: IMDB 5000 Movie Dataset from Kaggle (~5000 movies, 28 features)
  - **Model Architecture**: Residual Neural Network (ResNet) implemented in PyTorch with residual connections and dropout regularization
  - **Optimization Goal**: Find optimal movie characteristics (Budget, Cast, Genre) that maximize Return on Investment (ROI = Gross/Budget) while maintaining:
    - IMDB Score between 9.0 and 10.0
    - Budget between $20M and $200M
  - **Key Results**: The NIO algorithm identified that a mid-range budget (~$110M) with high star power and specific genre combinations yields optimal ROI (9.09x) while maintaining critical acclaim
  - Location: `NIO/`

### 🔢 Tensor Operations
Foundational notebooks for understanding data manipulation and tensor math.

- **NumPy & PyTorch**: Introductory notebooks covering the basics of NumPy arrays and PyTorch tensor operations, essential for any Deep Learning workflow.
  - Location: `tensor-operations/`

## 🛠️ Tech Stack & Tools with

- **Languages**: Python
- **Deep Learning Frameworks**: PyTorch
- **Machine Learning Libraries**: XGBoost, Scikit-Learn (implied)
- **Data Manipulation**: NumPy, Pandas
- **Model Exchange**: ONNX (Open Neural Network Exchange)
- **Environment**: Jupyter Notebooks

## 🚧 Status

This project is in a **Work-In-Progress** state. I am constantly learning and adding new implementations, including but not limited to:
- Computer Vision models (CNNs, ViTs)
- NLP architectures (Transformers, RNNs)
- Reinforcement Learning algorithms
- More advanced GAN architectures

Feel free to explore the code!
