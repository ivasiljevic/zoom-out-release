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
require 'matpay.lua'
torch.setdefaulttensortype('torch.FloatTensor')

mean_pix = {103.939, 116.779, 123.68};

fixedimsize = 256

require('initSBatchNormalization.lua')


bn = nn.initSBatchNormalization(3)

a = torch.Tensor(1,3,2,2):fill(0)

bn.affine = false
bn.running_mean:fill(2)
bn.running_var:fill(4)
bn.save_mean:fill(2)
bn.save_std:fill(4)
bn:parameters()[1]:fill(1) -- gamma
bn:parameters()[2]:fill(0) -- beta

bn:forward(a)