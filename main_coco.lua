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
dofile "data/load_coco.lua"
dofile "utils/preprocess.lua"
dofile "train.lua"
dofile "val.lua"
dofile "model/zoomout_construct.lua"
dofile "model/zoomout_classifier.lua"
require('utils/Replicatedynamic.lua')
require("utils/initSBatchNormalization.lua")
matio = require 'matio'
---------------------------------------------
--Paths to models and normalization tensors--
---------------------------------------------
MODEL_FILE = '/share/data/vision-greg/mlfeatsdata/caffe_temptest/examples/imagenet/VGG_ILSVRC_16_layers_fullconv.caffemodel'
CONFIG_FILE = '/home-nfs/reza/features/caffe_weighted/caffe/modelzoo/VGG_ILSVRC_16_layers_fulconv_N3.prototxt'
DATA_PATH = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/voc12-rand-all-val_GT.mat'
CLASSIFIER_PATH = '/share/data/vision-greg/mlfeatsdata/CV_Course/spatialcls_104epochs_normalizedmanual_deconv.t7'
--MODEL_PATH = "/share/data/vision-greg/ivas/model.net"
MODEL_PATH = 'model.net'
NORM_PATH = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/convglobalmeanstd.t7'
IMAGE_PATH = "/share/data/vision-greg/mlfeatsdata/CV_Course/voc12-val_GT.mat"
--------------------------------------------
--Setting up zoomout feature extractor------
--------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-new_model',1,"Create new model or load pre-trained")
cmd:option('-global', 1, "Include global features")
cmd:option('-origstride', 4, "Specify zoomout model stride")
cmd:option('-nlabels', 21,"Specify number of GT labels")
cmd:option('-nhiddenunits', 1000,"Specify number of hidden units")
cmd:option('-inputsize', 8320, "Specify feature dimension of input to classifier")
cmd:option('-downsample',4,"Set level of downsampling")
cmd:option('-train_val',1,"1 if training, 0 if validating")
cmd:option('-freeze', 0, "Freeze feature extractor")
cmd:option('-lr',1e-3, "Learning Rate")
cmd:option('-wd',0,"Weight Decay")
cmd:option('-momentum',0.9,"Momentum")
cmd:option('-dampening',0.0,"Dampening")
cmd:option('-lrd',0,"Learning Rate Decay")
cmd:option('-epoch',3,"Number of Epochs")
cmd:option('-batchsize',1,"Batch size for SGD")
cmd:option('-fixedh',256,"Fixed height for preprocessing")
cmd:option('-fixedw',336,"Fixed width for preprocessing")   
cmd:option('-coord', 1, "Add coordinate tensors to model")
cmd:text()

opt = cmd:parse(arg)
new_model = opt.new_model
inputsize = opt.inputsize
global = opt.global
origstride = opt.origstride
nlabels = opt.nlabels
nhiddenunits = opt.nhiddenunits
downsample = opt.downsample
fixedimh = opt.fixedh
fixedimw = opt.fixedw

-- Load Dataset
--train_data, train_gt = load_data(DATA_PATH)

mean_pix = {103.939, 116.779, 123.68} -- mean over PASCAL VOC dataset
zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30,36} --don't train classifier

--------------------------------------------
-----Set up the Classifier network---------
--------------------------------------------
if new_model==1 then
    net = loadcaffe.load(CONFIG_FILE, MODEL_FILE)
    classifier = zoomout_classifier(origstride,nlabels,nhiddenunits,inputsize)
    loadedmeanstd = torch.load(NORM_PATH)
    
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

    model = zoomout_construct(net,classifier,downsample,zlayers,global)
    print("New model constructed")
end


--Load pre-trained classifier 
if new_model==0 then
    model = torch.load(MODEL_PATH) 
    print("Pretrained model loaded.")
end

--Set up the loss function
criterion = cudnn.SpatialCrossEntropyCriterion()
criterion:cuda()
model:cuda()

--Cleanup
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

--optimMethod = optim.sgd
optimMethod = optim.adam
------------------------------------------
----------Zoomout Training----------------
------------------------------------------
if opt.train_val == 1 then
    print("Starting to train..")
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
            print(jj)
            local index = rand[jj]
            --local im = image.load(train_data[index])
            local im=image.load("/share/data/vision-greg/coco/train2014/"..string.sub(im_path[index],1,-5)..".jpg")
            if im:size()[1] < 3 then goto continue end
            local temp_gt=matio.load("/share/data/vision-greg/coco/gt-voc/"..im_path[index])
            local loaded = temp_gt.groundTruth[1].Segmentation
            --Do random flips 
            --if torch.randperm(2)[2]==2 then
            --    im_proc = preprocess(image.hflip(im:clone()),mean_pix,fixedimh,fixedimw)
            --    im_proc = im_proc:reshape(1, im_proc:size()[1],im_proc:size()[2],im_proc:size()[3])
            --    gt_proc = preprocess_gt_deconv(image.hflip(loaded:clone()),fixedimh, fixedimw)
            --    gt_proc = gt_proc:resize(1,gt_proc:size()[1],gt_proc:size()[2])
            --else      
                im_proc = preprocess(im,mean_pix,fixedimh,fixedimw)
                im_proc = im_proc:reshape(1, im_proc:size()[1],im_proc:size()[2],im_proc:size()[3])
                gt_proc = preprocess_gt_deconv(loaded,fixedimh, fixedimw)
                gt_proc = gt_proc:resize(1,gt_proc:size()[1],gt_proc:size()[2])
            --end
            print(im_proc:size())
            print(gt_proc:size())
            im = nil
            loaded = nil

            train(model, im_proc:cuda(), gt_proc:cuda())

            im_proc = nil
            gt_proc = nil
            collectgarbage()
            ::continue::
        end
    torch.save("model.net",model)
    end
end


