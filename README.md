#Zoomout Semantic Segmentation 

##Introduction

Zoomout is an architecture for semantic segmentation.  It takes advantage of a pre-trained classifier to compute zoomout features, which are intermediate feature maps that have been upsampled to be the same size as the input.  Then, these features are fed into a classifier that outputs a posterior distribution over every pixel in the image.

For details, please consult the ArXiv paper here - http://arxiv.org/pdf/1412.0774.pdf 

## Data Processing
The data processing scripts are dataset.lua and preprocess.lua (located in the train subdirectory). The included data processing scripts are most relevant for the PASCAL VOC and MS COCO datasets, but can be modified for any dataset. preprocess.lua resizes the images to be a fixed, pre-chosen height and width, and subtracts the mean color vector (provided for VOC).   

## Pre-trained classifier
Our zoomout architecture was built on top of the VGG-16 model, but you can pass the zoomout constructor any appropriate model (e.g. ResNet).  (Short description of zoomout construct, unsure of what we have to change for now).

## Building the Zoomout model
There are several options available to building the zoomout architecture. Specify origstride, nhiddenunits, downsample, fixedimh, fixedwid, zlayers
Replicatedynamic
Custom spatialbatchnorm
talk about precomputing the mean/stdev vectors for the spatialbatchnorm
## Accuracy
Without a CRF, the current architecture achieves 70% mean intersection-over-union (MIOU) on the PASCAL VOC 2012 challenge. Adding a dense CRF on top (include ref) increases accuracy to 72.XX%.

(Include a pretrained model, size considerations?) 

## Training 
Currently only works on batchsize = 1,should allow arbitrary batch size
Mention that it's currently doing horizontal flips. 

Training from scratch with VGG-16 as the classifier base and training end-to-end, can get over 65% MIOU in 3 epochs.

## Validation
Evaluate using matlab scripts.  Saving .mat files.
 
