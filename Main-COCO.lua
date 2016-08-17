require 'nngraph'
require 'torch'   
require 'image'   
require 'nn'      
require 'cunn'
require 'mattorch'
require 'cudnn'
matio = require 'matio'
require 'loadcaffe'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim' 
dofile "dataset.lua"
dofile "preprocess.lua"
dofile "train.lua"
dofile "val.lua"
dofile "coordinate.lua"
dofile "zoomoutconstruct.lua"
dofile "zoomoutclassifier.lua"
require('Replicatedynamic.lua')
require("initSBatchNormalization.lua")
---------------------------------------------
--Paths to models and normalization tensors--
---------------------------------------------

model_file='/share/data/vision-greg/mlfeatsdata/caffe_temptest/examples/imagenet/VGG_ILSVRC_16_layers_fullconv.caffemodel';
config_file='/home-nfs/reza/features/caffe_weighted/caffe/modelzoo/VGG_ILSVRC_16_layers_fulconv_N3.prototxt';
train_file = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/voc12-rand-all-val_GT.mat'
classifier_path = '/share/data/vision-greg/mlfeatsdata/CV_Course/spatialcls_104epochs_normalizedmanual_deconv.t7'
--model_path = "model.net"
--model_path = "/share/data/vision-greg/ivas/model.net"
normalize_path = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/convglobalmeanstd.t7'
image_path = "/share/data/vision-greg/mlfeatsdata/CV_Course/voc12-val_GT.mat"

--------------------------------------------
--Setting up zoomout feature extractor------
--------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-new_model',1,"Create new model or load pre-trained")
cmd:option('-epoch_num',1,"Check what epoch number")
cmd:text()
opt = cmd:parse(arg)

new_model = opt.new_model
train_data, train_gt = load_data(train_file)
mean_pix = {103.939, 116.779, 123.68};
fixedimh = 256
fixedwid = 336
fixedimsize = 256
downsample = 4
zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30,36}
--zlayers = {2,7,9,12,14,16,19,21,23,26,28,30,36} --don't train classifier
global = 1
origstride =4
nlabels = 21  
nhiddenunits = 1000
inputsize = 8320
--train or val?
val = 0 
training = 1
freeze_zoomout = 0

if new_model==1 then
net = loadcaffe.load(config_file, model_file)
--------------------------------------------
-----Set up the Classifier network---------
--------------------------------------------
classifier = torch.load(classifier_path)
loadedmeanstd = torch.load(normalize_path)

meanx = loadedmeanstd[1]
stdx = loadedmeanstd[2]

for i=1, stdx:size()[1] do
stdx[i]=stdx[i]^2
    if stdx[i]==0 then
    stdx[i]=1;
    end
end
--------------------------------------
--Batch Norm for Mean/Var Subtraction
--------------------------------------
if inputsize == 8320 then
batch_norm = nn.initSBatchNormalization(inputsize)
batch_norm.running_mean=meanx
batch_norm.affine=false
batch_norm.running_var=stdx
batch_norm.save_mean = meanx
batch_norm.save_var = stdx
batch_norm:parameters()[1]:fill(1)
batch_norm:parameters()[2]:fill(0)
classifier:insert(batch_norm,1)
end
--[[
classifier:get(1).weight:fill(1)
--classifier:get(1).bias:fill(0)
batch_norm = nil
for tt=1,inputsize do 
   -- classifier:get(1).weight[tt]=stdx[tt]
    classifier:get(1).bias[tt] = -meanx[tt]
    classifier:get(1).weight[tt] = 1/stdx[tt]
 --   classifier:get(1).bias[tt] = -meanx[tt]
end
--]]
model = zoomoutconstruct(net,classifier,downsample,zlayers,global)
end
if new_model == 0 then 
print("Loading old model")
model = torch.load(model_path) end

criterion = cudnn.SpatialCrossEntropyCriterion()
criterion:cuda()
model:cuda()

--------------------------------------------
---------------Validation------------------
--------------------------------------------

if val==1 then
    model:evaluate()
    s,sgt = load_data(image_path)
    validate(model)
end

classifier = nil
net = nil
loadedmeanstd = nil
meanx = nil
stdx = nil
batch_norm = nil
--------------------------------------------
--Training setup
--------------------------------------------
if model then
   parameters,gradParameters = model:getParameters()
end

optimState = {
  learningRate = 1e-4,
  weightDecay =1e-4,
  momentum = 0.9,
  dampening = 0.0,
  learningRateDecay =1e-4 
}

optimMethod = optim.sgd
--------------------
--Zoomout Training--
--------------------
dofile "datatest.lua"
if training == 1 then
model:training()
print("Training model.")
if freeze_zoomout then
for i, m in ipairs(model.modules) do
   if torch.type(m):find('Convolution') then
      m.accGradParameters = function() end
      m.updateParameters = function() end
end
end
end

possible_epoch = 35000
--if epoch_num == 0 then
rand0 = torch.randperm(66843)
--torch.save("index.txt",rand0)
--count = 1
--else
--rand0 = torch.load("index.txt")
--count = possible_epoch+1
--end

count = 1
for jj=1, 25000 do
    collectgarbage() 
    index0  = rand0[count]
    --if 1 then 
    --if torch.random(1,2) ==1 then 
    im = image.load(coco_dir..string.sub(temp[index0],1,-5)..".jpg")
    if im:size()[1] < 3 then goto continue end
--    if im:size()[1] < 3 then
--    im = torch.expand(im,3,im:size()[2],im:size()[3])
--    end
    temp_gt = matio.load("/share/data/vision-greg/coco/gt-voc/"..temp[index0])
    ground = temp_gt.groundTruth[1].Segmentation
--    ground[ground:eq(0)] = 21
    im_proc = preprocess(im,mean_pix)
    im_proc = im_proc:reshape(1, im_proc:size()[1],im_proc:size()[2],im_proc:size()[3])
    gt_proc = preprocess_gt_deconv(ground)
    gt_proc = gt_proc:resize(1,gt_proc:size()[1],gt_proc:size()[2])
temp_gt = nil
ground = nil
--[[
    else
    im = image.load(train_data[index1])
    loaded = matio.load(train_gt[index1]) -- be carefull, Transpose!! 
    im_proc = preprocess(im,mean_pix,fixedimsize)
    im_proc = im_proc:reshape(1, im_proc:size()[1],im_proc:size()[2],im_proc:size()[3])
    gt_proc = preprocess_gt_deconv(loaded.GT,fixedimsize)
    gt_proc = gt_proc:resize(1,gt_proc:size()[1],gt_proc:size()[2])
--]]
   -- end 
count = count + 1
print(im_proc:size())
print(gt_proc:size())

im = nil

    train(model, im_proc:cuda(), gt_proc:cuda())
    im_proc = nil
    gt_proc = nil
    collectgarbage()
   ::continue::
end
end
model:evaluate()
s,sgt = load_data(image_path)
validate(model)
--torch.save("model.net",model)

