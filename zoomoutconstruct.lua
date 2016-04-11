

function zoomoutconstruct(net,downsample)
mean_pix = {103.939, 116.779, 123.68};
--Fully conv
net:cuda() 
net:evaluate();	
kW = torch.Tensor(18):fill(3)
dW = torch.Tensor(18):fill(1)
stride = torch.Tensor(18)
sz = torch.Tensor(18)

sz[1] = 1
stride[1] = 1

--Moving up from layer m-1 to m:
kW[4] =2 
dW[4] =2
kW[7] =2 
dW[7] =2
kW[11] =2 
dW[11] =2
kW[15] =2 
dW[15] =2
for i =2, 18 do
sz[i] = sz[i-1] + (kW[i]-1)*stride[i-1]
stride[i] = stride[i-1]*dW[i]
end




downsample = 4
scale = 0
kersize = torch.Tensor(18)
stridesize = torch.Tensor(18)
padsize = torch.Tensor(18)

for i =1,18 do
kersize[i] = sz[i]/stride[i]
stridesize[i] = downsample/stride[i]
padsize[i] = torch.ceil(kersize[i]/3)
end

kersize = torch.Tensor(18):fill(1) -- let's fix the ker size for all layers. 
padsize = torch.Tensor(18):fill(0)
-- first conv and mapping layers
i=2
iminput = nn.Identity()()
C11 = net:get(1)(iminput)
C11 = net:get(2)(C11) -- relu output
S11 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C11)

i=3
C12 = net:get(3)(C11)
C12 = net:get(4)(C12)
-- suppix mapping
S12 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C12)


i=5
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C21 = net:get(5)(C12)  -- avg pool
C21 = net:get(6)(C21) 
C21 = net:get(7)(C21) -- relu
-- suppix mapping
S21 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C21)
if scale > 1 then
	S21 = nn.SpatialUpSamplingNearest(scale):cuda()(S21)
end




i=6
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C22 = net:get(8)(C21) 
C22 = net:get(9)(C22) -- relu
-- suppix mapping
S22 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C22)
if scale > 1 then
	S22 = nn.SpatialUpSamplingNearest(scale):cuda()(S22)
end


i=8
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C31 = net:get(10)(C22)  -- avg pool
C31 = net:get(11)(C31) 
C31 = net:get(12)(C31) -- relu
-- suppix mapping
S31 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C31)
if scale > 1 then
	S31 = nn.SpatialUpSamplingNearest(scale):cuda()(S31)
end



i=9
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C32 = net:get(13)(C31) 
C32 = net:get(14)(C32) -- relu
-- suppix mapping
S32 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C32)
if scale > 1 then
	S32 = nn.SpatialUpSamplingNearest(scale):cuda()(S32)
end



i=10
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C33 = net:get(15)(C32)
C33 = net:get(16)(C33) -- relu
-- suppix mapping
S33 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C33)
if scale > 1 then
	S33 = nn.SpatialUpSamplingNearest(scale):cuda()(S33)
end




i=12
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C41 = net:get(17)(C33)  -- avg pool
C41 = net:get(18)(C41)
C41 = net:get(19)(C41) -- relu
-- suppix mapping
S41 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C41)
if scale > 1 then
	S41 = nn.SpatialUpSamplingNearest(scale):cuda()(S41)
end


i=13
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C42 = net:get(20)(C41)
C42 = net:get(21)(C42) -- relu
-- suppix mapping
S42 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C42)
if scale > 1 then
	S42 = nn.SpatialUpSamplingNearest(scale):cuda()(S42)
end



i=14
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C43 = net:get(22)(C42)
C43 = net:get(23)(C43) -- relu
-- suppix mapping
S43 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C43)
if scale > 1 then
	S43 = nn.SpatialUpSamplingNearest(scale):cuda()(S43)
end



i=16
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C51 = net:get(24)(C43)  -- avg pool
C51 = net:get(25)(C51) 
C51 = net:get(26)(C51) -- relu
-- suppix mapping
S51 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C51)
if scale > 1 then
	S51 = nn.SpatialUpSamplingNearest(scale):cuda()(S51)
end

i=17
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C52 = net:get(27)(C51) 
C52 = net:get(28)(C52) -- relu
-- suppix mapping
S52 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C52)
if scale > 1 then
	S52 = nn.SpatialUpSamplingNearest(scale):cuda()(S52)
end

i=18
if stridesize[i] < 1 then
  scale = 1/stridesize[i]
  stridesize[i] = 1
end
C53 = net:get(29)(C52) 
C53 = net:get(30)(C53) -- relu
-- suppix mapping
S53 = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C53)
if scale > 1 then
	S53 = nn.SpatialUpSamplingNearest(scale):cuda()(S53)
end



C61 = net:get(31)(C53)  -- avg pool
C61 = net:get(32)(C61) 
--suppixinput19 = padmod:get(1)(suppixinput18)
C61 = net:get(33)(C61) -- relu
-- suppix mapping
--S61 = nn.suppixavgpool_table()({C61,suppixinput19})

C71 = net:get(34)(C61) -- dropout
C71 = net:get(35)(C71) 
C71 = net:get(36)(C71) -- relu

globalfeats = nn.Max(4):cuda()(C71)
globalfeats = nn.Max(3):cuda()(globalfeats)




repl1 = nn.Replicate(S53:size()[4],1):cuda()(globalfeats)
--temp = repl:forward(concatfeats[2])();
--repl2 = nn.Replicate(concatfeats[1]:size()[3],2):cuda()
--globfeats = repl:forward(temp):transpose(3,1):transpose(2,4);
--Join = nn.JoinTable(2):cuda():forward({concatfeats[1], globfeats});

Join = nn.JoinTable(2):cuda()({S11, S12, S21, S22, S31, S32, S33, S41, S42, S43, S51,  S52, S53})
zoomout_model = nn.gModule({iminput}, {Join,globalfeats,repl1})



net = nil

return zoomout_model
end
