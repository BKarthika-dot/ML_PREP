📌 Project Title

Image Classification with TensorFlow & CNN

📖 Overview

This project builds a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images into multiple categories.

The model was trained on a custom dataset using data pipelines built with tf.data, and performance was evaluated using accuracy, loss curves, and a confusion matrix.

🧠 Model Architecture

Input Layer

Rescaling (1./255)

Convolutional Layers + ReLU

MaxPooling Layers

Dense Layers

Softmax Output Layer

Regularization techniques used:

EarlyStopping

ModelCheckpoint (best model saving)

Class weights (to address imbalance)

📊 Training Results
Metric	Value
Final Training Accuracy	~45%
Final Validation Accuracy	~43%
Best Validation Loss	~1.50
Training Curves

Accuracy and validation accuracy tracked across epochs

Loss and validation loss monitored to detect overfitting

📉 Evaluation

Confusion Matrix generated on test set

Per-class performance analyzed

Best model restored using ModelCheckpoint


⚙️ How To Run

Clone the repository

Install dependencies:

pip install tensorflow matplotlib seaborn numpy


Run the notebook from top to bottom

🛠️ Tools Used

Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

🚀 What I Learned

Building CNN architectures

Working with tf.data pipelines

Handling class imbalance

Using callbacks (EarlyStopping & ModelCheckpoint)

Evaluating models beyond accuracy

📌 Future Improvements

Transfer learning (ResNet / EfficientNet)

Hyperparameter tuning

Cross-validation

