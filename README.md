Meal Assessment System
This project is a Meal Assessment System that uses a pre-trained MobileNet model to classify food items from images and estimate their calorie content. The system is built as a web application using Flask, allowing users to upload images of meals for analysis. The project includes a trained machine learning model, a Flask web application, and supporting files for food data and model training.

Project Structure
project/
├── app.py                          # Main Flask application script
├── models/
│   └── keras_models/
│       └── model-mobilenet-RMSprop0.0002-008-0.995584-0.711503.h5  # Pre-trained MobileNet model
├── static/
│   └── food_list/
│       ├── food_list.json       # JSON file mapping food classes to labels
│       └── Food_calories.csv   # CSV file containing calorie information for food items
├── templates/
│   └── Meal_Assessment_System.html  # HTML template for the web interface
├── model_training/
│   ├── f1.ipynb                 # Jupyter notebook for model training
│   └── README                  # README file specific to the model training process
├── upload_image/                   # Directory containing sample images for testing the model

Prerequisites
To run this project, ensure you have the following installed:

Python 
pip (Python package manager)
Virtualenv (optional, but recommended for isolated environments)

Required Python packages:

Flask
TensorFlow
NumPy
Pandas
Pillow (PIL)

You can install the dependencies using:
pip install flask tensorflow numpy pandas pillow

Setup Instructions

Clone or Download the Project:

Clone the repository or download and extract the project folder to your local machine.


Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:

Navigate to the project directory and install the required packages:pip install -r requirements.txt

Note: If a requirements.txt file is not provided, manually install the packages listed in the Prerequisites section.


Directory Setup:

Ensure the upload_image/ directory exists and is writable, as this is where uploaded images will be temporarily stored.
Verify that the models/keras_models/ directory contains the pre-trained model file (model-mobilenet-RMSprop0.0002-008-0.995584-0.711503.h5).
Confirm that static/food_list/ contains food_list.json and Food_calories.csv.


Run the Flask Application:

From the project root directory, run:python app.py


The Flask server will start, typically on http://127.0.0.1:5000.


Access the Web Interface:

Open a web browser and navigate to http://127.0.0.1:5000.
Use the web interface (Meal_Assessment_System.html) to upload an image of a meal.



How to Use

Web Interface:

The web application is accessible via the Flask server.
Upload an image of a meal using the provided interface in Meal_Assessment_System.html.
The system will process the image using the pre-trained MobileNet model, classify the food item, and estimate its calorie content based on data from Food_calories.csv.


Model Details:

The pre-trained model (model-mobilenet-RMSprop0.0002-008-0.995584-0.711503.h5) is a MobileNet model trained with the RMSprop optimizer (learning rate 0.0002) for 8 epochs, achieving a training accuracy of approximately 99.56% and validation accuracy of 71.15%.
The model uses food_list.json to map predicted class indices to food item names.
Calorie information is retrieved from Food_calories.csv.


Testing with Sample Images:

Place test images in the upload_image/ directory.
Use the web interface to upload these images and receive predictions.


Model Training:

The f1.ipynb notebook in the model_training/ directory contains the code used to train the MobileNet model.
Refer to the README file in the model_training/ directory for detailed instructions on training or retraining the model.



File Descriptions

app.py: The main Flask application script that handles the web server, image uploads, and model inference.
models/keras_models/model-mobilenet-RMSprop0.0002-008-0.995584-0.711503.h5: The pre-trained MobileNet model for food classification.
static/food_list/food_list.json: A JSON file mapping class indices to food item names used by the model.
static/food_list/Food_calories.csv: A CSV file containing calorie information for various food items.
templates/Meal_Assessment_System.html: The HTML template for the web interface where users can upload images and view results.
model_training/f1.ipynb: A Jupyter notebook containing the code for training the MobileNet model.
model_training/README: Documentation specific to the model training process.
upload_image/: A directory for storing uploaded or test images.

Notes

Ensure the upload_image/ directory has appropriate permissions for reading and writing.
The model expects images in a format compatible with TensorFlow (e.g., JPEG or PNG).
If you encounter issues with the model or predictions, verify that the food_list.json and Food_calories.csv files are correctly formatted and align with the model’s output classes.
For retraining or modifying the model, refer to the f1.ipynb notebook and its accompanying README in the model_training/ directory.

Troubleshooting

Flask Server Not Starting: Ensure all dependencies are installed and the Python environment is correctly set up.
Model Loading Error: Verify that the .h5 model file is present in models/keras_models/ and is not corrupted.
Image Upload Issues: Check that the upload_image/ directory exists and is writable.
Incorrect Predictions: Ensure food_list.json and Food_calories.csv are correctly aligned with the model’s output classes.

License
This project is licensed under the MIT License. See the LICENSE file for details (if applicable).