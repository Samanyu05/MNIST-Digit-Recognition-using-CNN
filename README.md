# MNIST Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset.

## Project Overview

The goal of this project is to build a deep learning model that can classify handwritten digits from 0 to 9. The images are preprocessed and passed through convolutional and pooling layers before being classified using fully connected layers.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Convolutional Neural Network (CNN)
* MNIST Dataset

## Dataset

The project uses the MNIST handwritten digit dataset, which contains:

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels
* 10 classes: digits 0–9

The pixel values are normalized from 0–255 to 0–1 before training.

## Model Architecture

The CNN consists of:

1. Conv2D layer with 32 filters and a 3×3 kernel
2. MaxPooling2D layer with 2×2 pooling
3. Conv2D layer with 64 filters and a 3×3 kernel
4. MaxPooling2D layer with 2×2 pooling
5. Flatten layer
6. Dense layer with 128 neurons
7. Output Dense layer with 10 neurons using Softmax

## Training Configuration

* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy
* Metric: Accuracy
* Epochs: 5
* Validation Split: 10%

## Prediction

After training, the model predicts digits from the test dataset using:

```python
predictions = model.predict(x_test)
predicted_digit = np.argmax(predictions[0])
```

The predicted digit is then compared with the actual label.

## Model Saving

The trained model is saved as:

```text
digit_recognition_model.h5
```

This allows the trained model to be reused without training it again.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/mnist-digit-recognition.git
cd mnist-digit-recognition
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy matplotlib
```

### 3. Run the Program

```bash
python digit_recognition.py
```

The program will download the MNIST dataset automatically, preprocess the images, train the CNN, evaluate its performance, make a prediction, and save the trained model.

## Project Structure

```text
MNIST-Digit-Recognition/
│
├── digit_recognition.py
├── digit_recognition_model.h5
├── README.md
└── requirements.txt
```

## Results

The model evaluates its performance on the MNIST test dataset and prints the test accuracy:

```text
Test Accuracy: <accuracy>
```

It also displays an example test image along with the predicted digit.

## Learning Outcomes

Through this project, I learned:

* How Convolutional Neural Networks work for image classification
* Image preprocessing and normalization
* Building CNN models using TensorFlow and Keras
* Training and validating deep learning models
* Evaluating model accuracy
* Making predictions using a trained model
* Saving trained machine learning models

## Future Improvements

Possible improvements include:

* Increasing the number of training epochs
* Adding Dropout layers to reduce overfitting
* Adding Batch Normalization
* Using data augmentation
* Creating a web interface for handwritten digit recognition
* Deploying the model as an API

## Author

**Samanyu Rai**

BTech – Cyber Security

