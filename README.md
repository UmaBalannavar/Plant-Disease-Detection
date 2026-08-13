# Plant-Disease-Detection
I worked on this project in my final year internship at ispark learning solutions pvt limited.

# Plant Disease Detection

A deep learning-based plant disease detection system that classifies plant leaf images and predicts the corresponding disease. The project combines a trained image classification model with a web interface for easy image-based prediction.

## Project Overview

The system allows a user to upload an image of a plant leaf and receive a predicted disease classification.

The project explores deep learning-based image classification using **Convolutional Neural Networks (CNNs)** and frameworks including **TensorFlow** and **PyTorch**.

The application provides:

* Plant leaf image upload
* Image preprocessing for model inference
* Disease classification using a MobileNet-based model
* Prediction displayed through a web interface
* Visualization of convolutional feature maps for model interpretation

## Technologies Used

* Python
* TensorFlow
* PyTorch
* MobileNet
* Convolutional Neural Networks (CNN)
* Flask
* HTML / CSS / JavaScript

## Project Structure

```text
plant-disease-detection/
│
├── backend/          # Backend and model inference code
├── frontend/         # Web interface
├── requirements.txt  # Python dependencies
└── README.md
```

## How It Works

```text
Leaf Image
    ↓
Image Preprocessing
    ↓
MobileNet-based Deep Learning Model
    ↓
Feature Extraction
    ↓
Disease Classification
    ↓
Prediction displayed in Web Interface
```

## Features

### Plant Disease Classification

The user can upload a plant leaf image and the trained model predicts the corresponding disease class.

### CNN Feature Visualization

The application can display feature maps extracted from the initial convolutional layers, providing insight into the features learned by the model.

### Web Interface

A simple web interface allows users to upload leaf images and obtain predictions without directly interacting with the model code.

## Model

The project uses a **MobileNet-based architecture** for image classification.

Deep learning experimentation and implementation were carried out using **TensorFlow and PyTorch**.

The trained model is used during inference to classify uploaded plant leaf images.

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd plant-disease-detection
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the backend application according to the instructions provided in the backend directory.

Then open the frontend in a browser and upload a plant leaf image for analysis.

## Example Workflow

1. Upload a plant leaf image.
2. Click **Analyse Leaf**.
3. The image is preprocessed.
4. The MobileNet-based model performs inference.
5. The predicted disease is displayed.
6. Feature maps can be examined to understand intermediate CNN representations.

## Purpose

This project was developed to understand the practical application of **Deep Learning and Computer Vision** for image classification, including model training, image preprocessing, inference, and integration with a web application.

## Note

Model files and virtual-environment files are not included in the repository where they are unnecessary or too large. The required dependencies are provided through `requirements.txt`.

## Author

Uma Balannavar
