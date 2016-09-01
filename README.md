#Zoomout Semantic Segmentation 

##Introduction

Zoomout is a convolutional neural network architecture for semantic segmentation. It maps small image elements (pixels/superpixels)
to rich feature representations extracted from a sequence of nested regions of increasing extent. These regions are obtained by "zooming out" from the pixel
all the way to scene-level resolution. Then, these features are fed into a classifier that outputs a posterior distribution over every pixel in the image.  

For details, please consult the CVPR paper: http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Mostajabi_Feedforward_Semantic_Segmentation_2015_CVPR_paper.pdf 

![zoomout.png](https://bitbucket.org/repo/n8qkM7/images/3481931785-zoomout.png =50x100)

## Dependencies
Zoomout requires the following Torch libraries:

+ matio
+ cunn
+ nngraph
+ image
+ cudnn

## Data Processing
The pipeline requires few preprocessing steps.  Preprocess.lua contains functions to resize training images to a fixed width and height and subtract the mean color, and batch and ground truth versions of these. 
 Dataset.lua will load PASCAL VOC images, and we supply scripts in coco/ to also load MS COCO images. 

Include script to download PASCAL VOC images, or short instructions on how to get them set up?

## Building the Zoomout model
Our zoomout architecture was built on top of the VGG-16 model, but you can pass the zoomout constructor any appropriate model (e.g. ResNet).  We take a look at the arguments in turn:
`zoomoutconstruct(net,clsmodel,downsample,zlayers,global)`

+ clsmodel - this is where the classifier you are building zoomout from goes.

(Short description of zoomout construct, unsure of what we have to change for now).
There are several options available to building the zoomout architecture, and two custom layers:

+ origstride - was not used ?
+ inputsize - feature dimension of feature extractor
+ nhiddenunits - number of hidden units in classifier on top of feature extractor 
+ downsample - how much you want to downsample from input for zoomout features, spatial upsampling (by default bilinear interpolation) brings back to original size 
+ fixedimh, fixedwid - VOC images are large, so we resize to reduce spatial dimension. Depending on whether H > W or W > H
+ zlayers - specify the numbers of the layers from the pretrained classifier to use for zoomout
+ Replicatedynamic - custom replicate function
+ spatialbatchnorm - talk about precomputing the mean/stdev vectors for the spatialbatchnorm


## Pixel classifier
On top of the zoomout feature extractor, we have a pixel classifier that makes the final posterior probability predictions. The default classifier is a convnet with the following layers and hidden unit sizes:
describe CNN model 

## Accuracy
Without a CRF, the current architecture achieves 70% mean intersection-over-union (MIOU) on the PASCAL VOC 2012 challenge. Adding a dense CRF on top (include ref) increases accuracy to 72.XX%.
(Include a pretrained model, size considerations?) 

## Training 
The training steps are as follows (I need to flesh this out):

1. Load data
2. Construct zoomout feature extractor
3. Construct the pixel classifier
4. Run batch/online gradient descent

The script for training is included in train.lua, currently we are using stochastic gradient descent with momentum (0.9) but any optimizer should work (e.g. Adam).  The only data augmentation used is horizontal flips, each training image is flipped with probability 0.5. The script main.lua does the following: replicates our experimental setup, using VGG-16 as the base classifier and training end-to-end. After about 3-4 epochs, training from scratch should lead to a model with 66% MIOU. 

Issues:
Currently only works on batchsize = 1,should allow arbitrary batch size

## Validation
Val.lua calculates the predicted segmentation masks which are saved as .mat files. The matlab scripts mapDataSets.m and valIOU.m (need to rewrite these?) will then calculate MIOU over each class and average MIOU. Usage: after training and validation is complete, run matlab -r valIOU to see score. 

# To-Do
Example segmentations?
Weakly supervised version?
Images/visualizations