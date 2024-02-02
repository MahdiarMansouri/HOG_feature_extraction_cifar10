# Comparative Analysis of CIFAR-10 Classification Using ResNet-18 and HOG Features

## Overview
This repository hosts a project that compares two approaches for image classification on the CIFAR-10 dataset using the ResNet-18 model. The first approach trains the model on the original images, while the second uses Histogram of Oriented Gradients (HOG) features.

## Dataset
A subset of 1000 images from the CIFAR-10 dataset, which contains 60000 32x32 color images across 10 classes, is used for this experiment.

## Methodology
1. **ResNet-18 with Original Images**: 
   - Fine-tuning of a pre-trained ResNet-18 model on the CIFAR-10 subset.
   - Documentation of training specifics like epochs, learning rate, and optimization strategy.

2. **ResNet-18 with HOG Features**:
   - Extraction of HOG features from each channel of the subset images.
   - Creation of a new dataset based on these HOG features.
   - Training the ResNet-18 model on this HOG-based dataset.

## Comparing the Approaches
The performance of the model trained on original images is compared with the one trained on HOG features. This comparison is based on:
  - Accuracy metrics.
  - Confusion matrices to understand the classification performance for each class.
  - Visualizations of predictions to provide insights into model behavior.
    
## Results
  - The results section will detail the accuracy obtained by both models.
  - Discussion on how the HOG feature extraction impacts the model's performance.

## Visualizations
  - This section includes confusion matrices and prediction visualizations for both training approaches.
  - It provides a visual understanding of the model's performance and its capabilities in classifying images from the CIFAR-10 dataset.

## Requirements
  - PyTorch
  - torchvision
  - Other dependencies are listed in requirements.txt.
    
## Installation
Clone the repository and install the required packages:
    
    git clone https://github.com/MahdiarMansouri/HOG_feature_extraction_cifar10/blob/master/main.ipynb
    cd HOG_feature_extraction_cifar10
    pip install -r requirements.txt

