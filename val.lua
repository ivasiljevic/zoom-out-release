

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

mean_pix = {103.939, 116.779, 123.68};
fixedimh = 256
fixedwid = 336
fixedimsize = 256
inputsize = 8320

filepath = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/convglobalmeanstd.t7'
loadedmeanstd = torch.load(filepath)

meanx = loadedmeanstd[1]
stdx = loadedmeanstd[2]

for i=1, stdx:size()[1] do
    if stdx[i]==0 then
    stdx[i]=1;
    end
end

--Setting up the zoomout mode
model_file='/share/data/vision-greg/mlfeatsdata/caffe_temptest/examples/imagenet/VGG_ILSVRC_16_layers_fullconv.caffemodel';
config_file='/home-nfs/reza/features/caffe_weighted/caffe/modelzoo/VGG_ILSVRC_16_layers_fulconv_N3.prototxt';

net = loadcaffe.load(config_file, model_file)
downsample = 4
zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30,36}
global = 1

classifier = torch.load('/share/data/vision-greg/mlfeatsdata/CV_Course/spatialcls_104epochs_normalizedmanual_deconv.t7')

dim1 = 64
dim2 = 92

batch_norm = nn.SpatialBatchNormalization(inputsize)
--l1 = nn.View(inputsize,dim1*dim2)
--l2 = nn.Transpose({1,2})
--l3 = nn.Transpose({1,2})
--l4 = nn.View(inputsize,dim1,dim2)
--l5 = nn.View(1,inputsize,dim1,dim2)

--classifier:insert(l1,1)
--classifier:insert(l2,2)
classifier:insert(batch_norm,1)
--classifier:insert(l3,4)
--classifier:insert(l4,5)
--classifier:insert(l5,6)

classifier:get(1).weight = classifier:get(1).weight:fill(1)

for tt=1,inputsize do
classifier:get(1).weight[tt] = classifier:get(1).weight[tt]/stdx[tt]
classifier:get(1).bias[tt] = -meanx[tt]
end

model = zoomoutconstruct(net,classifier,downsample,zlayers,global)
model:evaluate()

dofile "dataset.lua"
filePath = "/share/data/vision-greg/mlfeatsdata/CV_Course/voc12-val_GT.mat"
s,sgt = load_data(filePath)


function validate(model)
    for k=1,#imlist do
        collectgarbage()

        local im = image.load(s[k])
        local im_proc_temp = preprocess(im,mean_pix)
        local im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
        im_proc[{{1},{},{},{}}] = im_proc_temp

        local pred = model:forward(im_proc:cuda())

        gt_temp = nil
        concatfeats = nil

        im_rescale = image.scale(pred[1]:float(),im:size()[3],im:size()[2],"bilinear")
        C,D = torch.max(im_rescale,1)
        gt = D:squeeze():double()
        matio.save('prediction/pred_'..k..'.mat',gt)
        xlua.progress(k,#imlist)
    end
end

validate(model)
