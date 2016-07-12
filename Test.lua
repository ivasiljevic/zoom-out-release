require 'nngraph'
require 'torch'   
require 'image'   
require 'nn'      
require 'cunn'
require 'mattorch'
require 'cudnn'
matio = require 'matio'
require 'loadcaffe'
require 'xlua' 
require 'optim' 
dofile "dataset.lua"
dofile "preprocess.lua"
dofile "train.lua"
dofile "val.lua"
dofile "zoomoutconstruct.lua"
dofile "zoomoutclassifier.lua"
dofile "coordinate.lua"
require('Replicatedynamic.lua')
require("initSBatchNormalization.lua")
---------------------------------------------
--Paths to models and normalization tensors--
---------------------------------------------
model_file='/share/data/vision-greg/mlfeatsdata/caffe_temptest/examples/imagenet/VGG_ILSVRC_16_layers_fullconv.caffemodel';
config_file='/home-nfs/reza/features/caffe_weighted/caffe/modelzoo/VGG_ILSVRC_16_layers_fulconv_N3.prototxt';
train_file = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/voc12-rand-all-val_GT.mat'
classifier_path = '/share/data/vision-greg/mlfeatsdata/CV_Course/spatialcls_104epochs_normalizedmanual_deconv.t7'
model_path = "/share/data/vision-greg/ivas/model.net"
normalize_path = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/convglobalmeanstd.t7'
image_path = "/share/data/vision-greg/mlfeatsdata/CV_Course/voc12-val_GT.mat"
--------------------------------------------
--Setting up zoomout feature extractor------
--------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-new_model',0,"Create new model or load pre-trained")
cmd:option('-global', 1, "Include global features")
cmd:option('-origstride', 4, "Specify zoomout model stride")
cmd:option('-nlabels', 21,"Specify number of GT labels")
cmd:option('-nhiddenunits', 1000,"Specify number of hidden units")
cmd:option('-inputsize', 8320, "Specify feature dimension of input to classifier")
cmd:option('-downsample',4,"Set level of downsampling")
cmd:option('-train_val',1,"1 if training, 0 if validating")
cmd:option('-freeze', 0, "Freeze feature extractor")
cmd:option('-lr',1e-4, "Learning Rate")
cmd:option('-wd',1e-4,"Weight Decay")
cmd:option('-momentum',0.9,"Momentum")
cmd:option('-dampening',0.0,"Dampening")
cmd:option('-lrd',1e-4,"Learning Rate Decay")
cmd:option('-epoch',1,"Number of Epochs")
cmd:option('-batchsize',1,"Batch size for SGD")
cmd:option('-fixedh',256,"Fixed height for preprocessing")
cmd:option('-fixedw',336,"Fixed width for preprocessing")   
cmd:option('-coord', 0, "Add coordinate tensors to model")
cmd:text()

opt = cmd:parse(arg)
inputsize = opt.inputsize
global = opt.global
origstride = opt.origstride
nlabels = opt.nlabels
nhiddenunits = opt.nhiddenunits
downsample = opt.downsample
fixedimh = opt.fixedh
fixedwid = opt.fixedw

--Load Dataset
train_data, train_gt = load_data(train_file)
mean_pix = {103.939, 116.779, 123.68} -- mean over PASCAL VOC dataset
fixedimsize = 256
--zlayers = {4,9,16,23,30,36} --don't train classifier
zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30,36} --don't train classifier
--------------------------------------------
-----Set up the Classifier network---------
--------------------------------------------
if opt.new_model then
    net = loadcaffe.load(config_file, model_file)
    classifier = zoomoutclassifier(origstride,nlabels,nhiddenunits,inputsize)
    loadedmeanstd = torch.load(normalize_path)

    meanx = loadedmeanstd[1][{{1,opt.inputsize}}] --normalize based on feature size
    stdx = loadedmeanstd[2][{{1,opt.inputsize}}]

    for i=1, stdx:size()[1] do
        stdx[i]=stdx[i]^2
        if stdx[i]==0 then
            stdx[i]=1;
        end
    end
--------------------------------------
--Batch Norm for Mean/Var Subtraction
---------------------------------------
        batch_norm = nn.initSBatchNormalization(inputsize)
        batch_norm.running_mean=meanx
        batch_norm.affine=false
        batch_norm.running_var=stdx
        batch_norm.save_mean = meanx
        batch_norm.save_var = stdx
        batch_norm:parameters()[1]:fill(1)
        batch_norm:parameters()[2]:fill(0)
        classifier:insert(batch_norm,1)

    model = zoomoutconstruct(net,classifier,downsample,zlayers,global)
else
--Load pre-trained classifier 
    model = torch.load(model_path) 
end
--Set up the loss function
criterion = cudnn.SpatialCrossEntropyCriterion()
criterion:cuda()
model:cuda()
--------------------------------------------
---------------Validation------------------
--------------------------------------------
if opt.train_val==0 then
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
-----------Training Setup-------------------
--------------------------------------------
if model then
   parameters,gradParameters = model:getParameters()
end

optimState = 
    {
    learningRate = opt.lr,
    weightDecay = opt.wd,
    momentum = opt.momentum,
    dampening = opt.dampening,
    learningRateDecay = opt.lrd
    }

optimMethod = optim.sgd
------------------------------------------
----------Zoomout Training----------------
------------------------------------------
if opt.train_val == 1 then

    model:training()

    if opt.freeze then
    --Zeros out gradients going through zoomout network, updating only classifier.
        for i, m in ipairs(model.modules) do
            if torch.type(m):find('Convolution') then
                m.accGradParameters = function() end
                m.updateParameters = function() end
            end
        end
    end

    for k=1,opt.epoch do
        rand = torch.randperm(numimages)

        for jj=1, numimages do
            collectgarbage() 
            local index = rand[jj]
            local im = image.load(train_data[index])
            local loaded = matio.load(train_gt[index])
            --Do random flips 
            if torch.randperm(2)[2]==2 then
                im_proc = preprocess(image.hflip(im:clone()),mean_pix,fixedimsize)
                im_proc = im_proc:reshape(1, im_proc:size()[1],im_proc:size()[2],im_proc:size()[3])
                gt_proc = preprocess_gt_deconv(image.hflip(loaded.GT:clone()),fixedimsize)
                gt_proc = gt_proc:resize(1,gt_proc:size()[1],gt_proc:size()[2])
            else      
                im_proc = preprocess(im,mean_pix,fixedimsize)
                im_proc = im_proc:reshape(1, im_proc:size()[1],im_proc:size()[2],im_proc:size()[3])
                gt_proc = preprocess_gt_deconv(loaded.GT,fixedimsize)
                gt_proc = gt_proc:resize(1,gt_proc:size()[1],gt_proc:size()[2])
            end

            im = nil
            loaded = nil

            train(model, im_proc:cuda(), gt_proc:cuda())

            im_proc = nil
            gt_proc = nil
            collectgarbage()
        end
    end
end

model:evaluate()
s,sgt = load_data(image_path)
validate(model)
--torch.save("model.net",model)
