

function zoomoutconstruct(net,clsmodel,downsample,zlayers,global)

clsmodel:cuda()
net:cuda() 
net:evaluate();	


stride = torch.Tensor(zlayers[#zlayers])
stride[1] = 1
lastdW = 1
for i =2, zlayers[#zlayers] do
	if net:get(i).dW then 
		lastdW = net:get(i).dW
		stride[i] = stride[i-1]*lastdW
	else -- for the layers that they don't have dW i.e. Relu
		stride[i] = stride[i-1]*lastdW		
	end
end



kersize = torch.Tensor(zlayers[#zlayers]):fill(1) -- let's fix the ker size for all layers. 
padsize = torch.Tensor(zlayers[#zlayers]):fill(0)
stridesize = torch.Tensor(zlayers[#zlayers])

for i =1, zlayers[#zlayers] do
	stridesize[i] = downsample/stride[i]
end


scale = 0 
C = {}
S = {}
iminput = nn.Identity()()
downsamplefact = nn.Identity()()
C[1] = net:get(1)(iminput)
counter = 1
if global == 1 then
	numlayers = zlayers[#zlayers] - 1
	numzoomlayers = #zlayers -1
else
	numlayers = zlayers[#zlayers]
	numzoomlayers = #zlayers
end

for i = 2, numlayers do
	C[i] =  net:get(i)(C[i-1])
	if i == zlayers[counter] then
		if stridesize[i] < 1 then
			scale = 1/stridesize[i]
			stridesize[i] = 1
		end
		S[counter] = nn.SpatialMaxPooling(kersize[i],kersize[i],stridesize[i],stridesize[i],padsize[i],padsize[i]):cuda()(C[i])
		if scale > 1 then
			S[counter] = nn.SpatialUpSamplingNearest(scale):cuda()(S[counter])
		end
		counter = counter + 1
	end
end

Join = S[1]
for i = 2, numzoomlayers do
	Join = nn.JoinTable(2):cuda()({Join,S[i]})
end


if global == 1 then
	C[numlayers+1] = net:get(numlayers+1)(C[numlayers])
	globalfeats = nn.Max(4):cuda()(C[numlayers+1])
	globalfeats = nn.Max(3):cuda()(globalfeats)
	repl = nn.Replicatedynamic(1,4):cuda()({globalfeats,iminput})
	imtranspose = nn.Transpose({3, 4}):cuda()(iminput)
	repl = nn.Replicatedynamic(2,4):cuda()({repl,imtranspose})
	output = nn.Transpose({3, 1},{2,4}):cuda()(repl)
	Join = nn.JoinTable(2):cuda()({Join,output})
end

output = clsmodel(Join)
zoomout_model = nn.gModule({iminput}, {output})

return zoomout_model
end
