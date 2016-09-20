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
dofile "../utils/preprocess.lua"
dofile "../utils/Replicatedynamic.lua"
dofile "../utils/initSBatchNormalization.lua"

MODEL_PATH = "/share/data/vision-greg/ivas/model.net" 

mean_pix = {103.939, 116.779, 123.68}
fixed_h = 256
fixed_w = 336

model = torch.load(MODEL_PATH) 
sample_image = image.load("02.jpg")

model:evaluate()
--Preprocess Image
im_proc_temp = preprocess(sample_image,mean_pix,fixed_h, fixed_w)
im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
im_proc[{{1},{},{},{}}] = im_proc_temp

--Predict posterior distribution
pred = model:forward(im_proc:cuda())

--Upsample and find argmax for segmentation
im_rescale = image.scale(pred[1]:float(),sample_image:size()[3],sample_image:size()[2],"bilinear")
C,D = torch.max(im_rescale,1)

--Save segmentation
label = D:squeeze():double()/21
image.save("mask.jpg", label)
