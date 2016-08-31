#Zoomout Semantic Segmentation 

##Introduction

Zoomout is a convolutional neural network architecture for semantic segmentation.  It takes advantage of a pre-trained classifier to compute zoomout features, which are intermediate feature maps that have been upsampled to be the same size as the input.  Then, these features are fed into a classifier that outputs a posterior distribution over every pixel in the image.  
For details, please consult the ArXiv paper here - http://arxiv.org/pdf/1412.0774.pdf 

## Dependencies
+ matio
+ cunn
+ nngraph
+ image
+ cudnn
## Data Processing
The data processing scripts are dataset.lua and preprocess.lua (located in the train subdirectory). The included data processing scripts are most relevant for the PASCAL VOC and MS COCO datasets, but can be modified for any dataset. preprocess.lua contains functions for the following: resizing images to a fixed height/width, color mean subtraction, and batch versions of those functions. 
Dataset.lua loads VOC images and ground truth. 

## Pre-trained classifier
Our zoomout architecture was built on top of the VGG-16 model, but you can pass the zoomout constructor any appropriate model (e.g. ResNet).  (Short description of zoomout construct, unsure of what we have to change for now).

## Building the Zoomout model
There are several options available to building the zoomout architecture, and two custom layers:

+ origstride - was not used ?
+ inputsize - feature dimension of feature extractor
+ nhiddenunits - number of hidden units in classifier on top of feature extractor 
+ downsample - how much you want to downsample from input for zoomout features, spatial upsampling (by default bilinear interpolation) brings back to original size 
+ fixedimh, fixedwid - VOC images are large, so we resize to reduce spatial dimension. Depending on whether H > W or W > H
+ zlayers - specify the numbers of the layers from the pretrained classifier to use for zoomout
+ Replicatedynamic - custom replicate function
+ spatialbatchnorm - talk about precomputing the mean/stdev vectors for the spatialbatchnorm

## Accuracy
Without a CRF, the current architecture achieves 70% mean intersection-over-union (MIOU) on the PASCAL VOC 2012 challenge. Adding a dense CRF on top (include ref) increases accuracy to 72.XX%.
(Include a pretrained model, size considerations?) 

## Training 
Step-by-step list?
1.
2. 
3. 
4. 
The script for training is included in train.lua, currently we are using stochastic gradient descent with momentum (0.9) but any optimizer should work (e.g. Adam).  The only data augmentation used is horizontal flips, each training image is flipped with probability 0.5. The script main.lua does the following: replicates our experimental setup, using VGG-16 as the base classifier and training end-to-end. After about 3-4 epochs, training from scratch should lead to a model with 66% MIOU. 

Issues:
Currently only works on batchsize = 1,should allow arbitrary batch size

## Validation
Val.lua calculates the predicted segmentation masks which are saved as .mat files. The matlab scripts mapDataSets.m and valIOU.m (need to rewrite these?) will then calculate MIOU over each class and average MIOU. Usage: after training and validation is complete, run matlab -r valIOU to see score. 

# To-Do
Example segmentations?
Weakly supervised version?
Images/visualizations 
