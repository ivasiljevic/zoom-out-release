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
dofile "../model/zoomout_construct.lua"



opt = nil
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'features', 'subdirectory to save features in')
   cmd:option('-MODEL_FILE', 'VGG_ILSVRC_16_layers_fullconv.caffemodel') --Directory where the pre-trained net such as VGG-16 is located
   cmd:option('-CONFIG_FILE', 'VGG_ILSVRC_16_layers_fulconv_N3.prototxt') -- Directory where config file is located
   cmd:option('-image', '02.jpg') -- Directory where the image is located
   cmd:text()
   opt = cmd:parse(arg or {})
end



mean_pix = {103.939, 116.779, 123.68}
-- Setting the input to a fixed size 
-- It is not neccessary for the input to be a fixed size we did it here just for simplicity and saving memory
fixed_h = 256 
fixed_w = 336
downsample = 4 -- Output downsample factor 
zlayers = {2,4,7,9,12,14,16,19,21,23,26,28,30,36} -- Subset of conv layers that we want to get their features
global = 1 -- Having the global feature in the feature set
clsmodel = nil -- No classifier, just zoomout features

-- Load pre-trained net such as VGG-16
net = loadcaffe.load(opt.CONFIG_FILE, opt.MODEL_FILE)
-- Create zoomout model from pre-trained classifier
zoomout_model = zoomout_construct(net, clsmodel, downsample, zlayers, global)

sample_image = image.load(opt.image)

-- Preprocess image
im_proc_temp = preprocess(sample_image,mean_pix,fixed_h, fixed_w)
-- Set the input in a batch mode i.e. batch size = 1
im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
im_proc[{{1},{},{},{}}] = im_proc_temp

-- Extract zoomout features
zoomout_model:evaluate()
zoomout_feats = zoomout_model:forward(im_proc)

print("Zoomout feature dimension: ", zoomout_feats:size())

filename = paths.concat(opt.save, 'Zfeatures.t7')
os.execute('mkdir -p ' .. sys.dirname(filename))
print('==> saving features to '..filename)
torch.save(filename, zoomout_feats:float())
