require 'nngraph'
require 'torch'   
require 'image'   
require 'nn'      
require 'cunn'
require 'mattorch'
matio = require 'matio'
require 'loadcaffe'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim' 
dofile "zoomoutsample.lua"
dofile "dataset.lua"

filepath = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/convglobalmeanstd.t7'

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
train_data,train_gt = load_data(filePath)
samp = sparse_zoomout_features(zoomout_model,train_data,train_gt,meanx,stdx)
--torch.save("sampling/sampfeats.t7",sampfeats)

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

    local cord_x = coordinate_x(Join:size()[3],Join:size()[4])
    local cord_y = coordinate_y(Join:size()[3],Join:size()[4])

    local x = cord_x:resize(1,1,Join:size()[3],Join:size()[4])
    local y = cord_y:resize(1,1,Join:size()[3],Join:size()[4])

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
