--Constructs classifier model.

function zoomout_classifier(origstride,nlabels,nhiddenunits,inputsize)
    local mod = nn.Sequential() 
    mod:add(nn.SpatialConvolutionMM(inputsize, nhiddenunits, 1, 1))
    mod:add(nn.ReLU(true))
    mod:add(nn.Dropout(0.5))
    mod:add(nn.SpatialConvolutionMM(nhiddenunits, nhiddenunits, 1, 1))
    mod:add(nn.ReLU(true))
    mod:add(nn.Dropout(0.5))
    mod:add(nn.SpatialConvolutionMM(nhiddenunits,21, 1, 1))
    --mod:add(nn.SpatialFullConvolution(nlabels,nlabels,origstride*2,origstride*2,origstride,origstride,origstride/2,origstride/2))
    mod:add(nn.SpatialFullConvolution(21,nlabels,origstride*2,origstride*2,origstride,origstride,origstride/2,origstride/2))
    local W = Bilinearkernel(origstride*2,nlabels,21)  -- initailization to bilinear upsampling
    mod.modules[8].weight= W
    mod.modules[8].bias:fill(0);
    return mod
end
