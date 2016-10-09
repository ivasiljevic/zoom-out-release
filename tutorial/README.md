## Segmentation Inference Tutorial

The segmentation.lua file runs inference on a single image given a pretrained zoomout model. The final segmentation mask is saved as an image.

th segmentation.lua -save 'subdirectory to save the output in' -MODEL_FILE 'pretrained model' -image 'input image'

## Zoomout Feature Extraction Tutorial

The zoomout_feature.lua file extracts zoomout features for a single image. We load the pretrained classifier (such as VGG-16) and construct the zoomout feature extractor with zoomout_construct, with no pixelwise classifier on top. 

After we extract the zoomout features, we check the dimension of the zoomout tensor.

th zoomout_feature.lua -save 'subdirectory to save features in'  -MODEL_FILE 'Caffe model such as VGG-16' -CONFIG_FILE 'Caffe configuration file' -image 'input image'
 
