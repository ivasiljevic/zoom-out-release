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
dofile "zoomoutconstruct.lua"
dofile "zoomoutclassifier.lua"
require('Replicatedynamic.lua')

--Setting up the zoomout mode
model_file='/share/data/vision-greg/mlfeatsdata/caffe_temptest/examples/imagenet/VGG_ILSVRC_16_layers_fullconv.caffemodel';
config_file='/home-nfs/reza/features/caffe_weighted/caffe/modelzoo/VGG_ILSVRC_16_layers_fulconv_N3.prototxt';

net = loadcaffe.load(config_file, model_file)

filePath = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/voc12-rand-all-val_GT.mat'
train_data, train_gt = load_data(filePath)
mean_pix = {103.939, 116.779, 123.68};
fixedimh = 256
fixedwid = 336
fixedimsize = 256
downsample = 4
zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30,36}
--zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30}
global = 1
origstride =4
nlabels = 21  
nhiddenunits = 1000
inputsize = 8320

--Set up the zoomout network
--]classifier = zoomoutclassifier(origstride,nlabels,nhiddenunits,inputsize)
--classifier = torch.load("results/model.net")
--The issue:
classifier = torch.load('/share/data/vision-greg/mlfeatsdata/CV_Course/spatialcls_104epochs_normalizedmanual_deconv.t7')
batch_norm = nn.BatchNormalization(inputsize)
l1 = nn.View(inputsize,64*84)
l2 = nn.Transpose({1,2})
l3 = nn.Transpose({1,2})
l4 = nn.View(inputsize,64,84)
l5 = nn.View(1,inputsize,64,84)

classifier:insert(l1,1)
classifier:insert(l2,2)
classifier:insert(batch_norm,3)
classifier:insert(l3,4)
classifier:insert(l4,5)
classifier:insert(l5,6)


filepath = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/convglobalmeanstd.t7'
loadedmeanstd = torch.load(filepath)

meanx = loadedmeanstd[1]
stdx = loadedmeanstd[2]

for i=1, stdx:size()[1] do
    if stdx[i]==0 then
    stdx[i]=1;
    end
end

classifier:get(3).weight = classifier:get(3).weight:fill(1)

for tt=1,inputsize do
classifier:get(3).weight[tt] = classifier:get(3).weight[tt]/stdx[tt]
classifier:get(3).bias[tt] = -meanx[tt]
end

zoomout_model = zoomoutconstruct(net,classifier,downsample,zlayers,global)
criterion = cudnn.SpatialCrossEntropyCriterion()
criterion:cuda()

--[[
W = Bilinearkernel(origstride*2,nlabels,nlabels)  -- initailization to bilinear upsampling
zoomout_model.modules[76]:get(8).weight= W
zoomout_model.modules[76]:get(8).bias:fill(0);
--]]
zoomout_model = zoomout_model:cuda()
model = zoomout_model
--dofile "val.lua"
--validate(model)



--]]
classifier = nil
net = nil


-- classes
classes = {'1','2','3','4','5','6','7','8','9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if zoomout_model then
   parameters,gradParameters = zoomout_model:getParameters()
end


optimState = nil
optimState = {
  learningRate = 0.0001,
  weightDecay = 0.0,
  momentum = 0.9,
  dampening = 0.0,
  learningRateDecay = 1e-3
}
optimMethod = optim.sgd

filepath = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/convglobalmeanstd.t7'

fixedimsize = 256
fixedwid = 336
fixedimh = 256

loadedmeanstd = torch.load(filepath)

meanx = loadedmeanstd[1]
stdx = loadedmeanstd[2]

for i=1, stdx:size()[1] do
    if stdx[i]==0 then
    stdx[i]=1;
    end
end 

------------------
--Sampling Model--
------------------
--[[
dofile "temp_zoomout.lua"
pixels = 100
train_data,train_gt = load_data(filePath)
samp = sparse_zoomout_features(zoomout_model,train_data,train_gt,pixels,meanx,stdx)
torch.save("sampling/sampfeats.t7",samp)
--]]
--------------------
--Zoomout Training--
--------------------

batchsize = 1 
datasetlabels = torch.Tensor(batchsize,fixedimh,fixedwid)
im_proc = torch.Tensor(batchsize,3,fixedimh,fixedwid)
rand = torch.randperm(numimages)


for jj=1, numimages do
    collectgarbage()
    for i=1,batchsize do
    index = 1--rand[jj]
    local im = image.load(train_data[index])
    local loaded = matio.load(train_gt[index]) -- be carefull, Transpose!!

    if torch.randperm(2)[2]==2 then
    im_proc_temp = preprocess(image.hflip(im:clone()),mean_pix)
    im_proc = torch.Tensor(batchsize,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
    im_proc[{{i},{},{},{}}] = im_proc_temp
    gt_temp = preprocess_gt_deconv(image.hflip(loaded.GT:clone()))
    gt_proc = torch.Tensor(batchsize,gt_temp:size()[1],gt_temp:size()[2])
    gt_proc[{{i},{},{}}] =  gt_temp
    else
    im_proc_temp = preprocess(im,mean_pix)
    im_proc = torch.Tensor(batchsize,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
    im_proc[{{i},{},{},{}}] = im_proc_temp

    gt_temp = preprocess_gt_deconv(loaded.GT)
    gt_proc = torch.Tensor(batchsize,gt_temp:size()[1],gt_temp:size()[2])
    gt_proc[{{i},{},{}}] =  gt_temp
    end
    end 
    train(zoomout_model, im_proc:cuda(), gt_proc:cuda())

    gt_temp = nil
    repl = nil
    temp = nil
    im_proc_temp = nil
    concatfeats = nil
    im_proc = nil
    Join = nil
    gt_proc = nil
    collectgarbage()
end

