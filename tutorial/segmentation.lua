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



opt = nil
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'output', 'subdirectory to save features in')
   cmd:option('-MODEL_FILE', 'zoomout_model.net') --Directory where the pre-trained net such as VGG-16 is located
   cmd:option('-image', '02.jpg') -- Directory where the image is located
   cmd:text()
   opt = cmd:parse(arg or {})
end


mean_pix = {103.939, 116.779, 123.68}
-- It is not neccessary for the input to be a fixed size we did it here just for simplicity and saving memory
fixed_h = 256 
fixed_w = 336

model = torch.load(opt.MODEL_FILE) 
sample_image = image.load(opt.image)

model:cuda()
model:evaluate()
--Preprocess Image
im_proc_temp = preprocess(sample_image,mean_pix,fixed_h, fixed_w)
-- Set the input in a batch mode i.e. batch size = 1
im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
im_proc[{{1},{},{},{}}] = im_proc_temp

--Predict posterior distribution
pred = model:forward(im_proc:cuda())

--Upsample and find argmax for segmentation
im_rescale = image.scale(pred[1]:float(),sample_image:size()[3],sample_image:size()[2],"bilinear")
heatmap, label = torch.max(im_rescale,1)

--Save segmentation
label = label:squeeze():double()/21

filename = paths.concat(opt.save, 'segmentationmask.jpg')
os.execute('mkdir -p ' .. sys.dirname(filename))
print('==> saving output to '..filename)
image.save(filename, label)
