# Fashion-MNIST-CNN-Classifier
A deep learning project to classify fashion images using Convolutional Neural Networks (CNNs) on the Fashion MNIST dataset.
👗 Conquering Fashion MNIST with CNNs using Computer Vision
   A deep learning project that classifies clothing images using CNNs, trained on the Fashion MNIST dataset. Achieved up to 99.1% accuracy, improving over traditional ML methods like SVM.

📌 Project Description

  This project explores the use of Convolutional Neural Networks (CNNs) to classify images of fashion items such as T-shirts, shoes, bags, etc. The goal is to outperform classical ML models (like SVM with 89.7%) using a deep learning approach.



 🧠 Technologies Used

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib


🗃️ Dataset

- Fashion MNIST (preloaded in Keras)
- 60,000 training and 10,000 testing grayscale images (28x28 pixels)
- 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot



 🏗️ Model Architecture

text
Input (28x28x1)
↓
Conv2D (64 filters, 5x5) + ReLU + MaxPooling
↓
Conv2D (128 filters, 5x5) + ReLU + MaxPooling
↓
Conv2D (256 filters, 5x5) + ReLU + MaxPooling
↓
Flatten → Dense(256, ReLU) → Dense(10, Softmax)
