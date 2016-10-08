#Zoomout Semantic Segmentation 

##Introduction

Zoomout is a convolutional neural network architecture for semantic segmentation. It maps small image elements (pixels or superpixels)
to rich feature representations extracted from a sequence of nested regions of increasing extent. These regions are obtained by "zooming out" from the pixel
all the way to scene-level resolution. Then, these features are fed into a classifier that outputs a posterior distribution over every pixel in the image.  

For details, please consult and cite the CVPR paper: http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf 

@inproceedings{mostajabi2015feedforward,

  title={Feedforward semantic segmentation with zoom-out features},
  
  author={Mostajabi, Mohammadreza and Yadollahpour, Payman and Shakhnarovich, Gregory},
  
  booktitle={CVPR},
  
  year={2015}
  
}

![zoomout.png](https://bitbucket.org/repo/n8qkM7/images/3302094990-zoomout.png)

## Dependencies
Zoomout requires the following Torch libraries:

+ cunn
+ nngraph
+ image
+ cudnn

## Data Processing
The zoomout pipeline requires few image pre-preprocessing steps.  
Preprocess.lua contains functions to resize training images to a fixed width and height (and to subtract the mean color), including batch and ground truth versions of these functions.
Dataset.lua will load PASCAL VOC images, and we supply scripts in coco/ to also load MS COCO images. 

## Building the Zoomout model
Our zoomout architecture was built on top of the VGG-16 model, but you can pass the zoomout constructor any appropriate model (e.g. ResNet).  There are several options available to building the zoomout architecture:

`zoomoutconstruct(net,clsmodel,downsample,zlayers,global)`

+ clsmodel - the classifier the zoomout model is built from
+ inputsize - feature dimension of feature extractor
+ nhiddenunits - number of hidden units in classifier on top of feature extractor 
+ downsample - how much you want to downsample from input for zoomout features, spatial upsampling (by default bilinear interpolation) brings back to original size 
+ fixedimh, fixedwid - VOC images are large, so we resize to reduce spatial dimension. Depending on whether H > W or W > H
+ zlayers - specify the numbers of the layers from the pretrained classifier to use for zoomout
+ Replicatedynamic - custom replicate function
+ spatialbatchnorm - custom layer for mean/stdev normalization for the zoomout features, precomputed mean/stdev vectors included

## Pixel classifier
On top of the zoomout feature extractor, we have a pixel classifier (zoomoutclassifier.lua) that makes the final posterior probability predictions. The default classifier is a 4-layer convolutional neural network.  The last layer of the classifier is a bilinear interpolation so that the label predictions match the spatial size of the ground truth. 

## Accuracy
Without a CRF, the current architecture achieves 70% mean intersection-over-union (MIOU) on the PASCAL VOC 2012 challenge. Adding a dense CRF on top (include ref) increases accuracy to 72%.

## Training 
The training steps are as follows:

1. Load data (VOC or COCO)
2. Construct zoomout feature extractor (from e.g. VGG-16)
3. Construct the pixel classifier
4. Run batch/online gradient descent

The script for training is included in train.lua, currently we are using stochastic gradient descent with momentum (0.9) but any optimizer should work (e.g. Adam).  The only data augmentation used is horizontal flips, each training image is flipped with probability 0.5. The script main.lua does the following: replicates our experimental setup, using VGG-16 as the base classifier and training end-to-end. After about 3-4 epochs, training from scratch should lead to a model with 66% MIOU. 

## Pretrained model

http://ttic.uchicago.edu/~mostajabi/files/zoomout_model.net


