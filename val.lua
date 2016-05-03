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

dofile "dataset.lua"
filePath = "/share/data/vision-greg/mlfeatsdata/CV_Course/voc12-val_GT.mat"
s,sgt = load_data(filePath)
model = torch.load("results/model.net")
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
