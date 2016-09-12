require 'nngraph'
require 'torch'   
require 'image'   
require 'nn'      
require 'cunn'
require 'mattorch'
require 'cudnn'
require 'loadcaffe'
require 'xlua' 
require 'optim' 
dofile "../train/preprocess.lua"
dofile "../train/Replicatedynamic.lua"
dofile "../train/initSBatchNormalization.lua"
dofile "../train/zoomoutconstruct.lua"

model_file='/share/data/vision-greg/mlfeatsdata/caffe_temptest/examples/imagenet/VGG_ILSVRC_16_layers_fullconv.caffemodel';
config_file='/home-nfs/reza/features/caffe_weighted/caffe/modelzoo/VGG_ILSVRC_16_layers_fulconv_N3.prototxt';

mean_pix = {103.939, 116.779, 123.68}
fixed_h = 256
fixed_w = 336
downsample = 4
zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30,36}
global = 1
clsmodel = nil -- No classifier, just zoomout features

net = loadcaffe.load(config_file, model_file)
zoomout_model = zoomoutconstruct(net, clsmodel, downsample, zlayers, global)

sample_image = image.load("02.jpg")

im = sample_image
im_proc_temp = preprocess(sample_image,mean_pix,fixed_h, fixed_w)
im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
im_proc[{{1},{},{},{}}] = im_proc_temp
zoomout_feats = zoomout_model:forward(im_proc)
print(zoomout_feats:size())

