Food Image Classification with MobileNetV2 :-
This project implements a deep learning model for classifying food images using the Kaggle Food-101 dataset. The model is built using MobileNetV2 as a pre-trained base, fine-tuned for multi-class classification across 25 food categories. The training was performed on Google Colab with free GPU resources, utilizing a strategy to maximize GPU usage by switching between multiple Colab accounts to handle the large dataset.
Table of Contents


Overview
This project trains a convolutional neural network (CNN) to classify images from the Food-101 dataset, which contains 101,000 images across 101 food categories. Due to the large dataset size, the code processes a subset of 25 categories (configurable based on the folder structure in image_dir). The model leverages transfer learning with MobileNetV2, pre-trained on ImageNet, and adds custom dense layers for classification. The training process was distributed across multiple Google Colab accounts to overcome free GPU quota limits, with the model saved and loaded between sessions.

Dataset
The Food-101 dataset is sourced from Kaggle and contains 101,000 images across 101 food categories, with 1,000 images per category. For this project, a subset of 25 categories was used (based on the available folder structure).

Source: Kaggle Food-101 Dataset
Directory Structure: Images are organized in subfolders named after their respective food categories (e.g., chana_masala, pizza).
Preprocessing: Images are resized to 224x224 pixels, rescaled to [0,1], and augmented (rotation, zoom, flips, etc.) for training.

Requirements
To run the code, install the following dependencies:
pip install tensorflow pandas numpy matplotlib seaborn pillow scikit-learn opencv-python

Additional requirements:

Python 
Google Colab environment with GPU support (for training)
Kaggle Food-101 dataset downloaded and accessible in the working directory or Google Drive
Google Drive mounted for saving/loading models

Project Structure
The project consists of a single Python script (f1.ipynb) that handles the entire workflow:

f1.ipynb: Main script containing data preprocessing, model training, evaluation, and prediction.
Dataset: Expected in a directory (image_dir) with subfolders for each food category.
Saved Model: The trained model is saved to Google Drive (/content/drive/MyDrive/f1) and loaded for evaluation/prediction.
Output: Training/validation accuracy and loss plots, test accuracy, and sample predictions.

Training Process
The training process involves:

Data Loading:
Images are loaded recursively from image_dir using pathlib.
Labels are extracted from folder names.
Data is split into training (80%) and testing (20%) sets using train_test_split.


Data Augmentation:
Training data is augmented using ImageDataGenerator with rotation, zoom, flips, and shifts.
Test data is only rescaled.


Model Training:
MobileNetV2 is used as the base model with pre-trained ImageNet weights (frozen layers).
Custom dense layers are added for classification.
The model is trained for 30 epochs with the Adam optimizer and categorical cross-entropy loss.


Model Saving/Loading:
The trained model is saved to /content/drive/MyDrive/f1.
The model is loaded for evaluation and prediction.


Evaluation:
Training and validation accuracy/loss are plotted.
Test accuracy is computed on the test dataset.


Prediction:
A sample image is classified, and the result is displayed with the image.



Model Architecture
The model is based on MobileNetV2 with the following structure:

Input: Images resized to 224x224x3 (RGB).
Base Model: MobileNetV2 (pre-trained on ImageNet, frozen layers).
Custom Layers:
GlobalAveragePooling2D
Dense(512, ReLU)
Dense(256, ReLU)
Dense(128, ReLU)
Dropout(0.5)
Dense(25, softmax)


Optimizer: Adam
Loss: Categorical Cross-Entropy
Metrics: Accuracy

Code Explanation
Below is a detailed breakdown of the key sections in f1.ipynb:
1. Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os.path
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential


Purpose: Imports necessary libraries for data manipulation (pandas, numpy), visualization (matplotlib, seaborn), file handling (pathlib, os), and deep learning (tensorflow, keras).

2. Mounting Google Drive
from google.colab import drive
drive.mount('/content/drive')


Purpose: Mounts Google Drive to save/load the model, enabling persistence across Colab sessions.

3. Data Loading and Preprocessing
image_dir = Path('')
filepath = list(image_dir.glob(r'**/*.jpg'))
label = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepath))
filepath = pd.Series(filepath, name='Filepath').astype(str)
label = pd.Series(label, name='Label')
image_df = pd.concat([filepath, label], axis=1).sample(frac=1.0, random_state=1).reset_index(drop=True)
train_df, test_df = train_test_split(image_df, test_size=0.20, random_state=43)


Purpose: 
Loads all .jpg images from image_dir recursively.
Extracts labels from folder names.
Creates a pandas DataFrame with file paths and labels.
Splits data into training (80%) and testing (20%) sets.



4. Data Augmentation
train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255, validation_split=0.2, rotation_range=20, width_shift_range=0.2,
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)
test_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_dataset = train_gen.flow_from_dataframe(dataframe=train_df, x_col='Filepath', y_col='Label', target_size=(224, 224), batch_size=32, color_mode='rgb', class_mode='categorical', shuffle=True, seed=42, subset='training')
validation_dataset = train_gen.flow_from_dataframe(dataframe=train_df, x_col='Filepath', y_col='Label', target_size=(224, 224), batch_size=32, color_mode='rgb', class_mode='categorical', shuffle=True, seed=42, subset='validation')
test_dataset = test_gen.flow_from_dataframe(dataframe=test_df, x_col='Filepath', y_col='Label', target_size=(224, 224), batch_size=32, color_mode='rgb', class_mode='categorical', shuffle=False)


Purpose:
Defines ImageDataGenerator for training (with augmentation) and testing (rescaling only).
Creates data generators for training, validation (20% of training data), and testing.
Images are resized to 224x224, batched (32), and processed in RGB format.



5. Class Indices
class_indices = train_dataset.class_indices
index_to_class = {v: k for k, v in class_indices.items()}


Purpose: Maps class indices to labels for prediction interpretation.

6. Model Definition
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(25, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


Purpose:
Loads MobileNetV2 with pre-trained ImageNet weights, excluding the top layer.
Freezes base model layers to use pre-trained features.
Adds custom layers: GlobalAveragePooling2D, three dense layers (512, 256, 128 units with ReLU), dropout (0.5), and a softmax layer for 25 classes.
Compiles the model with Adam optimizer and categorical cross-entropy loss.



7. Model Training
model_history = model.fit(train_dataset, validation_data=validation_dataset, epochs=30)


Purpose: Trains the model for 30 epochs, using training and validation datasets.

8. Model Saving
model.save('/content/drive/MyDrive/f1')


Purpose: Saves the trained model to Google Drive for reuse.

9. Model Loading and Evaluation
loaded_model = load_model("/content/drive/MyDrive/model-mobilenet-RMSprop0.0002-008-0.995584-0.711503")
acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')
loss, accuracy = loaded_model.evaluate(test_dataset)
print('Test accuracy :', accuracy)


Purpose:
Loads the saved model.
Plots training and validation accuracy.
Evaluates the model on the test dataset and prints test accuracy.



10. Prediction
loaded_model = load_model("/content/drive/MyDrive/model-mobilenet-RMSprop0.0002-008-0.995584-0.711503")
img_path = '/content/drive/MyDrive/1/chana_masala/22f9414273.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)
predictions = model.predict(img_array)
predicted_class_index = np.argmax(predictions)
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title(index_to_class[predicted_class_index])
plt.axis('off')
plt.show()


Purpose:
Loads the model (from /content/drive/MyDrive/model-mobilenet-RMSprop0.0002-008-0.995584-0.711503, which may need correction to match the saved path).
Loads and preprocesses a sample image.
Predicts the class and displays the image with the predicted label.



Usage

Setup:
Download the Food-101 dataset from Kaggle.
Upload the dataset to Google Drive or the Colab environment.
Update image_dir in f1.ipynb to point to the dataset directory.
Ensure Google Drive is mounted (drive.mount('/content/drive')).


Run the Script:python f1.ipynb


The script preprocesses data, trains the model, saves it, evaluates performance, and makes a sample prediction.


Prediction:
Update img_path in the prediction section to test a specific image.
The predicted class and image are displayed using Matplotlib.



Results

Training/Validation Accuracy: Plotted to visualize model performance over 30 epochs.
Test Accuracy: Computed on the test dataset and printed.
Sample Prediction: A sample image is classified, and the result is displayed with the image.

Training Across Multiple Accounts
To handle the large Food-101 dataset and Google Colab's free GPU quota limits, training was distributed across multiple Colab accounts:

Initial Training:
The model was trained on one Colab account until the GPU quota was exhausted.
The model was saved to Google Drive (/content/drive/MyDrive/f1).


Switching Accounts:
The saved model was loaded on a different Colab account with a fresh GPU quota.
Training resumed from the saved checkpoint.


Iteration:
This process was repeated across multiple accounts to complete the 30 epochs.


Final Model:
The final trained model was saved and used for evaluation and prediction.



Note: The code includes model saving (model.save) and loading (load_model) to support this process. Ensure the model path is consistent across accounts.
License
This project is licensed under the MIT License. See the LICENSE file for details.