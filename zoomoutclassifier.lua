--Constructs classifier model.

function zoomoutclassifier(mod,origstride,nlabels,nhiddenunits,inputsize)
    model = mod
    model:add(nn.SpatialConvolutionMM(inputsize, nhiddenunits, 1, 1))
    model:add(nn.ReLU(true))
    model:add(nn.Dropout(0.5))
    model:add(nn.SpatialConvolutionMM(nhiddenunits, nhiddenunits, 1, 1))
    model:add(nn.ReLU(true))
    model:add(nn.Dropout(0.5))
    model:add(nn.SpatialConvolutionMM(nhiddenunits, nlabels, 1, 1))
    model:add(nn.SpatialFullConvolution(nlabels,nlabels,origstride*2,origstride*2,origstride,origstride,origstride/2,origstride/2))
    W = Bilinearkernel(origstride*2,nlabels,nlabels)  -- initailization to bilinear upsampling
    model.modules[83].weight= W
    model.modules[83].bias:fill(0);

    criterion = cudnn.SpatialCrossEntropyCriterion()
    model:cuda()
    criterion:cuda()

    return model
end
