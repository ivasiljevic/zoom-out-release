## Segmentation Inference Tutorial

The segmentation.lua file runs inference on a single image given a pretrained zoomout model. First, the image is preprocessed, and after it's run through the zoomout model it has a dimension of W/4 x H/4. Thus, we must upsample (here using bilinear interpolation) before taking the argmax over posterior probabilities in order to find the predicted pixelwise labels.

The final segmentation mask is saved as an image.

## Zoomout Feature Extraction Tutorial

The zoomout_feature.lua file extracts zoomout features for a single image. We load the pretrained classifier (such as VGG-16) and construct the zoomout feature extractor with zoomout_construct, with no pixelwise classifier on top. 

After we extract the zoomout features, we check the dimension of the zoomout tensor.

 
