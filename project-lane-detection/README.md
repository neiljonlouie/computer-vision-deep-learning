# EE 298-F Project: Lane Detection

This project is an attempt at lane detection segmentation using an encoder-decoder architecture loosely based on ResNet (https://arxiv.org/pdf/1512.03385.pdf). The goal of this project is to create and train a lightweight model that could be easily ported onto mobile platforms. The data set used to train this model is the CULane Dataset from the Chinese University of Hong Kong (https://xingangpan.github.io/projects/CULane.html)

The code is written in Python 3, and assumes that the following packages (and its dependencies) are installed:
* Keras (with TensorFlow backend)
* OpenCV
* numpy

## Configuration
The file `main.py` contains all the configurations needed to perform testing or training.

## Training the Model
To train the model, run `python3 main.py train [--weights_file <weights file>] [--num_epochs <number of epochs>]`. If no weights file is provided, the model is trained from scratch. Otherwise, the model loads the provided checkpoint. The number of epochs is set to 20 by default.

For each pixel of the (resized) input image, the model produces a vector of probabilities (via softmax) of the pixel belonging to either the background (0) or one of four lane markings (1-4, with 1 being the leftmost lane boundary and 4 being the rightmost). Categorial cross-entropy is used as the loss function.

## Testing the Model
To detect the lanes of an image, run `python3 main.py test --weights_file <weights file> --input_file <path to image>`. A valid weights file (.hdf5) and the path to the image must be provided.

### Expected Output
After running the model, the post-processing steps include obtaining the predicted class given the class probabilities and annotating the original image with the detected lanes. From left to right, the detected lanes are colored blue, yellow, red, and cyan. The resulting annotated image is saved in `pred.png`.
