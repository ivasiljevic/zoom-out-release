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
model_path = "results/model.net"
normalize_path = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/convglobalmeanstd.t7'
image_path = "/share/data/vision-greg/mlfeatsdata/CV_Course/voc12-val_GT.mat"

--------------------------------------------
--Setting up zoomout feature extractor------
--------------------------------------------
new_model = 0
train_data, train_gt = load_data(train_file)
mean_pix = {103.939, 116.779, 123.68};
fixedimh = 256
fixedwid = 336
fixedimsize = 256
downsample = 4
zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30,36}
--zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30} --don't train classifier
global = 1
origstride =4
nlabels = 21  
nhiddenunits = 1000
inputsize = 8320
--train or val?
val = 0 
training = 1
if new_model then
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

batch_norm = nn.initSBatchNormalization(inputsize)
batch_norm.running_mean=meanx
batch_norm.affine=false
batch_norm.running_var=stdx
batch_norm.save_mean = meanx
batch_norm.save_var = stdx
batch_norm:parameters()[1]:fill(1)
batch_norm:parameters()[2]:fill(0)

classifier:insert(batch_norm,1)
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
if new_model == 0 then model = torch.load(model_path) end

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

--classes = {'1','2','3','4','5','6','7','8','9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'}
--confusion = optim.ConfusionMatrix(classes)

if model then
   parameters,gradParameters = model:getParameters()
end

optimState = {
  learningRate = 1e-4,
  weightDecay = 1e-4,--1e-5,
  momentum = 0.9,
  dampening = 0.0,
  learningRateDecay = 1e-4--1e-4
}
optimMethod = optim.sgd
--------------------
--Zoomout Training--
--------------------
if training == 1 then
model:training()

for k=1,4 do
batchsize = 1 
rand = torch.randperm(numimages)
for jj=1, numimages do
    collectgarbage() 
--    for i=1,batchsize do
        local index = rand[jj]
        local im = image.load(train_data[index])
        local loaded = matio.load(train_gt[index]) -- be carefull, Transpose!!
    
--freeMemory, totalMemory = cutorch.getMemoryUsage(1)
--        old_mem = freeMemory
        
   -- print(freeMemory, im:size())
    im_proc = preprocess(im,mean_pix,fixedimsize)
    im_proc = im_proc:reshape(1, im_proc:size()[1],im_proc:size()[2],im_proc:size()[3])
    gt_proc = preprocess_gt_deconv(loaded.GT,fixedimsize)
    gt_proc = gt_proc:resize(1,gt_proc:size()[1],gt_proc:size()[2])
    im =nil 
 --   print(im_proc:size())
    train(model, im_proc:cuda(), gt_proc:cuda())
    im_proc = nil
    gt_proc = nil
    loaded = nil
    collectgarbage()

--freeMemory, totalMemory = cutorch.getMemoryUsage(1)
--print(freeMemory)
--if jj > 10 and freeMemory < old_mem*0.5 then break 
--end
end
end
end
model:evaluate()
s,sgt = load_data(image_path)
validate(model)

--torch.save("model.net",model)
