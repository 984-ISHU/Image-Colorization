# Image Colorizer

## Overview

This project is an image colorizer web application that uses a deep learning model to colorize grayscale images. The application is built using Flask, a micro web framework in Python, and utilizes a Caffe model for the colorization process. The goal is to provide users with an easy-to-use interface for uploading grayscale images, processing them through a trained model, and downloading the colorized results.

## Project Structure

- `app.py`: Main Flask application file handling routes and logic.
- `templates/disp.html`: HTML template for displaying colorized images and providing download functionality.
- `static/`: Directory containing static assets such as CSS, JavaScript, and images.
- `uploads/`: Directory where uploaded grayscale images are stored.
- `downloads/`: Directory where colorized images are saved after processing.
- `model/`: Directory containing the Caffe model files.

## Caffe Model

> This repository doesn't contain `colorization_release_v2.caffemodel` file. Download it from [link](https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1) to your 'model' directory.

### Overview

Caffe is a deep learning framework developed by the Berkeley Vision and Learning Center (BVLC). It is known for its speed and modularity, making it suitable for a range of image processing tasks. The model used in this project is designed to colorize grayscale images.

### Model Components

1. **Prototxt File** (`colorization_deploy_v2.prototxt`):
   - Defines the architecture of the neural network.
   - Specifies the layers, their connections, and other model parameters.

2. **Caffemodel File** (`colorization_release_v2.caffemodel`):
   - Contains the pre-trained weights for the neural network.
   - The weights are obtained by training the model on a large dataset of color images.

4. **Numpy File** (`pts_in_hull.npy`):
   - Contains the cluster centers used by the model for colorization.
   - These cluster centers help in mapping grayscale values to color values.

### Image Colorization Process

1. **Image Preprocessing**:
   - Convert the input grayscale image to the LAB color space.
   - Extract the L channel (luminance) and resize the image for model input.

2. **Model Inference**:
   - Use the Caffe model to predict the AB channels (color) from the L channel.
   - Resize the predicted AB channels to match the dimensions of the original image.

3. **Image Postprocessing**:
   - Combine the L channel with the predicted AB channels.
   - Convert the LAB image back to BGR color space.
   - Save the colorized image to the specified directory.

## Website Functionality

### Pages

1. **Index Page (`index.html`)**:
   - The landing page of the website where users can upload grayscale images.
   - Provides a form for file upload.

2. **Display Page (`disp.html`)**:
   - Shows the original grayscale image and the colorized result side by side.
   - Includes a button for downloading the colorized image.

### Features

- **Image Upload**:
  - Users can upload grayscale images through a form on the index page.
  - The uploaded image is saved in the `uploads/` directory.

- **Image Processing**:
  - The uploaded image is processed using the Caffe model to produce a colorized version.
  - The colorized image is saved in the `downloads/` directory.

- **Image Download**:
  - Users can download the colorized image by clicking the download button.
  - After the download, users are redirected to the index page.

## Setup and Installation

### Requirements

- Python 3.x
- Flask
- OpenCV
- Numpy
- Caffe

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/984-ISHU/Image-Colorization.git
   cd Image-Colorization
