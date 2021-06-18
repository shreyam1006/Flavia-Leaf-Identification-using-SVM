# Flavia Leaf Identification Using SVM

Identification of plants through plant leaves on the basis of their shape, color and texture features using digital image processing techniques.

## Overview

The number of species of plants known by the botanists is ever increasing. In
our natural ecosystem there are so many plants that a normal person (not a
botanist) cannot distinguish between the different species of plants in our
surroundings. Say, any of us wants to get a plant that we saw somewhere in our
neighbourhood, to get the plant we need to identify which plant is it, where can
we get it etc. Plants and their species vary with respect to terrain, weather
conditions, soil type as well as odour and how it grows. However, most of the
identification of plants is based on the characteristics of its leaves- colour,
shape, size, form, flowers etc. Therefore, to determine the species of plants
studying the leaves is very essential.

Plant Leaf Identification is a system which is able to classify **32 different species of plants** on the basis of their leaves using digital image processing techniques. The images are first preprocessed and then their shape, color and texture based features are extracted from the processed image.

A dataset was created using the extracted features to train and test the model. The model used was **Support Vector Machine Classifier** and was able to classify with **90.05% accuracy**. 

## Dataset

The dataset used is [**Flavia leaves dataset**](http://flavia.sourceforge.net) which also has the breakpoints and the names mentioned for the leaves dataset

## Dependencies

* [Numpy](http://www.numpy.org)
* [Pandas](https://pandas.pydata.org)
* [OpenCV](https://opencv.org)
* [Matplotlib](https://matplotlib.org)
* [Scikit Learn](http://scikit-learn.org/)
* [Mahotas](http://mahotas.readthedocs.io/en/latest/)

It is recommended to use [Anaconda Python 3.6 distribution](https://www.anaconda.com) and using a `Jupyter Notebook`

## Instructions

* Create the following folders in the project root - 
  * `Flavia leaves dataset` : will contain Flavia dataset
  * `mobile captures` : will contain mobile captured leaf images for additional testing purposes

## Project structure

* [single_image_process_file.ipynb](single_image_process_file.ipynb) : contains exploration of preprocessing and feature extraction techniques by operating on a single image
* [background_subtract_camera_capture_leaf_file.ipynb](background_subtract_camera_capture_leaf_file.ipynb) : contains exploration of techniques to create a background subtraction function to remove background from mobile camera captured leaf images
* [classify_leaves_flavia.ipynb](Flavia%20py%20files/classify_leaves_flavia.ipynb) : uses extracted features as inputs to the model and classifies them using SVM classifier
* [preprocess_extract_dataset_flavia.ipynb](Flavia%20py%20files/preprocess_extract_dataset_flavia.ipynb) : contains create_dataset() function which performs image pre-processing and feature extraction on the dataset. The dataset is stored in `Flavia_features.csv`

## Methodology

### 1. Pre-processing

Steps involved in pre-processing of images are as follows:
1. Load the images to be used for training purposes from the Flavia dataset
available.
2. Convert all the RGB images into grayscale images for simplification.
3. Smoothing images using Gaussian filters.
4. Using Otsu’s thresholding method for adaptive thresholding of images.
5. Dealing with the holes using Morphological operations such as closing.
6. Edge extraction using contours.

### 2. Feature extraction

Many features were extracted from the images from pre-processing of images.
1. Shape- length, width, area, perimeter, circularity, rectangularity and aspect
ratio
2. Colour- mean and standard deviation of colour channels
3. Texture- entropy, correlation, contrast, inverse difference
  
### 3. Machine Learning Model building and testing

After feature extraction of the images we train our model providing them the
images of the Flavia leaves dataset using Support Vector Machine Learning
Classifier.
Later the featured scaling was done using StandardScalar. Moreover, parameter
tuning using GridSearchCV to find best parameters.
After that the model is tested on random pictures from the internet for better
results.

## Results:
The accuracy obtained by training the Support Vector Machine using Flavia
dataset online and testing it on a self obtained set of images came out to be
90.05% accurate.

## Conclusion:
We concluded using the Support Vector Machine approach to classify 32
different species of leaves. Experimental evidence shows that we were
successful in achieving more than 90% accuracy in classifying the different
species of leaves. In comparison to other algorithms, Support vector machines
are not only fast and much more accurate but also easy to implement. There is
always a scope of improving the accuracy for better performance of the model.
