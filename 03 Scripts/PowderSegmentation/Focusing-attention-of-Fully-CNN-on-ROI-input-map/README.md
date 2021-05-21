# Tensorflow 2.0 Files for Hierarchical semantic segmentation using modular convolutional neural networks ([here](../../../01%20Input/fromUoS/Papers/ImageSegmentation/hierarchical-semantic-segmentation-using-modular-convolutional-NN.pdf))

The original Tensorflow code for [Focusing attention of Fully convolutional neural networks on Region of interest (ROI) input map, using the valve filters method](https://github.com/sagieppel/Focusing-attention-of-Fully-convolutional-neural-networks-on-Region-of-interest-ROI-input-map-) was written for Tensorflow version 1. This folder contains the updated code so it can run in Tensorflow version 2.

The code was updated using the Tensorflow [tf_upgrade_v2](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/compatibility/tf_upgrade_v2.py) utility and some manual edits. The utility script changed the appropriate calls for version 1 tools and classes to tensorflow.compat.v1. Manual edits were for replacing functions removed from required libraries with their recommended equivalents.

This document is to note which ones have been changed and any new dependencies that have been added.

A full upgrade to the Tensorflow 2 idioms and classes is outside the scope of this project.

## Removed depdencies
  - scipy.misc
    + Previously used imread, imsave and imresize methods
    + Those methods have been removed from scipy since v1. See [here](https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.imread.html)

## New dependencies
  - [Matplotlib](https://matplotlib.org/)
    + matplotlib.pyplot.imread replaces scipy.misc.imread
  - [Pillow](https://pillow.readthedocs.io/en/stable/reference/Image.html)
    + Image class is used to convert the read-in image numpy arrays into valid images. 
    + Image.resize method replaces scipy.misc.resize.
    + Image class method imsave is also used to write results to folders
    
## Edited Scripts
  - [Data_Reader](Data_Reader.py)
    + Changed dependency on scipy.misc im* methods to appropriate Pillow methods
  - [BuildNetVgg16](BuildNetVgg16.py)
    + Changed dependency on scipy.misc im* methods to appropriate Pillow methods
    + Fixed syntax errors introduced from updated libraries
  - [Inference](Inference.py)
    + Fixed syntax errors introduced from updated libraries
    + Disabled eager execution
  - [TRAIN](TRAIN.py)
    + Fixed syntax errors introduced from updated libraries
    + Disabled eager execution
 
## New Scripts
  - [useCamera](useCamera.py)
    + Loads a model via checkpoint and feeds in images via a camera opened using OpenCV VideoCapture class. See [here](https://docs.opencv.org/4.1.0/d8/dfe/classcv_1_1VideoCapture.html).
    + Region of Interest (ROI) map is set as a filled bounding box centered on the image convering a fixed percentage of the image. It is applied to all images.
    + Results are displayed in three windows
      * Frame : Image retrieved from camera that is fed into the network
      * Overlay : Overlay of prediced labels ontop of the frame using OverLayLabelOnImage function from [Overlay](OverrlayLabelOnImage.py)
      * Labels : Predicted labels matrix returned from the model normalized to 0-255 so it can be displayed as an image.
    + Function runs continuously until user presses ESC on one of the OpenCV windows.
    + Returns the last labels matrix. Used for debugging.
