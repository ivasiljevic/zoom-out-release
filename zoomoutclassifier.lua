--Constructs classifier model.

function zoomoutclassifier(origstride,nlabels,nhiddenunits,inputsize)
    local model = nn.Sequential() 
    model:add(nn.SpatialConvolutionMM(inputsize, nhiddenunits, 1, 1))
    model:add(nn.ReLU(true))
    model:add(nn.Dropout(0.5))
    model:add(nn.SpatialConvolutionMM(nhiddenunits, nhiddenunits, 1, 1))
    model:add(nn.ReLU(true))
    model:add(nn.Dropout(0.5))
    model:add(nn.SpatialConvolutionMM(nhiddenunits, nlabels, 1, 1))
    model:add(nn.SpatialFullConvolution(nlabels,nlabels,origstride*2,origstride*2,origstride,origstride,origstride/2,origstride/2))
    W = Bilinearkernel(origstride*2,nlabels,nlabels)  -- initailization to bilinear upsampling
    model.modules[8].weight= W
    model.modules[8].bias:fill(0);

    criterion = cudnn.SpatialCrossEntropyCriterion()
    model:cuda()
    criterion:cuda()

    return model
end
