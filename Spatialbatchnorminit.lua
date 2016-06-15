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

	
require('initSBatchNormalization.lua')


bn = nn.initSBatchNormalization(3)

a = torch.Tensor(1,3,2,2):fill(0)

bn.affine = false

-- set these to mean and var vectors 
bn.running_mean:fill(2)  -- = meanx
bn.running_var:fill(4)  -- = stdx^2
bn.save_mean:fill(2) -- = meanx
bn.save_std:fill(4)  -- = stdx^2
-------------


bn:parameters()[1]:fill(1) -- gamma
bn:parameters()[2]:fill(0) -- beta

bn:forward(a)