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
global = 1
batchsize = 1 
origstride =4
nlabels = 21  
nhiddenunits = 1000
inputsize = 8320

datasetlabels = torch.Tensor(batchsize,fixedimh,fixedwid)
im_proc = torch.Tensor(batchsize,3,fixedimh,fixedwid)
local im = image.load(train_data[1])
im_proc_temp = preprocess(im:clone(),mean_pix)
im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])

zoomout_model = zoomoutconstruct(net,downsample,zlayers,global)
classifier = zoomoutclassifier(zoomout_model,origstride,nlabels,nhiddenunits,inputsize)

temp = classifier:forward(im_proc:cuda())

--[[
-- classes
classes = {'1','2','3','4','5','6','7','8','9','10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'}

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)


-- Log results to files
--trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
--testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters,gradParameters = model:getParameters()
end


optimState = nil
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  dampening = 0.0,
  learningRateDecay = 0
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

filePath = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/voc12-rand-all-val_GT.mat'
------------------
--Sampling Model--
------------------
dofile "temp_zoomout.lua"
pixels = 100
train_data,train_gt = load_data(filePath)
samp = sparse_zoomout_features(zoomout_model,train_data,train_gt,pixels,meanx,stdx)
torch.save("sampling/sampfeats.t7",samp)
--------------------
--Zoomout Training--
--------------------
--[[
batchsize = 1 
datasetlabels = torch.Tensor(batchsize,fixedimh,fixedwid)
im_proc = torch.Tensor(batchsize,3,fixedimh,fixedwid)
rand = torch.randperm(numimages)


for jj=1, numimages do
    collectgarbage()
    index = rand[jj]
    local im = image.load(s[index])
    local loaded = matio.load(sgt[index]) -- be carefull, Transpose!!

    if torch.randperm(2)[2]==2 then
    im_proc_temp = preprocess(image.hflip(im:clone()),mean_pix)
    im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
    im_proc[{{1},{},{},{}}] = im_proc_temp
    gt_temp = preprocess_gt_deconv(image.hflip(loaded.GT:clone()))
    gt_proc = torch.Tensor(1,gt_temp:size()[1],gt_temp:size()[2])
    gt_proc[{{1},{},{}}] =  gt_temp
    else
    im_proc_temp = preprocess(im,mean_pix)
    im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
    im_proc[{{1},{},{},{}}] = im_proc_temp

    gt_temp = preprocess_gt_deconv(loaded.GT)
    gt_proc = torch.Tensor(1,gt_temp:size()[1],gt_temp:size()[2])
    gt_proc[{{1},{},{}}] =  gt_temp
    end

    local concatfeats = zoomout_model:forward(im_proc:cuda())
    local repl = nn.Replicate(concatfeats[1]:size()[4],1):cuda()
    local temp = repl:forward(concatfeats[2]);
    local repl = nn.Replicate(concatfeats[1]:size()[3],2):cuda()
    local globfeats = repl:forward(temp):transpose(3,1):transpose(2,4)
    local Join = nn.JoinTable(2):cuda():forward({concatfeats[1], globfeats})

    for tt =1, Join:size()[2] do
        Join[{{},{tt},{},{}}]:add(-meanx[tt])
        Join[{{},{tt},{},{}}]:div(stdx[tt])
    end

    local dat = torch.cat(Join:float(),y,2)
    dat = torch.cat(dat, x,2)
    globfeats = nil
    gt_temp = nil
    repl = nil
    temp = nil
    im_proc_temp = nil
    concatfeats = nil
    im_proc = nil
    Join = nil

    collectgarbage()
    trsize = batchsize
    trainData = {
        data = dat,
        labels = gt_proc,
        size = function() return trsize end
        }
    dat = nil
    gt_proc = nil
    collectgarbage()
    train()
    trainData = nil
end
--]]
