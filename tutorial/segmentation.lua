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

mean_pix = {103.939, 116.779, 123.68}
fixed_h = 256
fixed_w = 336

model_path = "/share/data/vision-greg/ivas/model.net" 
model = torch.load(model_path) 

sample_image = image.load("02.jpg")

model:evaluate()
im = sample_image
im_proc_temp = preprocess(sample_image,mean_pix,fixed_h, fixed_w)
im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
im_proc[{{1},{},{},{}}] = im_proc_temp
pred = model:forward(im_proc:cuda())

im_rescale = image.scale(pred[1]:float(),im:size()[3],im:size()[2],"bilinear")
C,D = torch.max(im_rescale,1)
label = D:squeeze():double()/21
image.save("mask.jpg", label)
