--Defines the preprocessing functions needed for zoomout model.

function preprocess(im, mean_pix)
    imheight = fixedimh
    imwidth = fixedwid
    if im:size()[2] > im:size()[3] then
        imwidth = fixedimh
        imheight = fixedwid
    end
    local im3 = image.scale(im:float(),imwidth,imheight,'bilinear')*255
     -- RGB2BGR
    local im4 = im3:clone()
    im4[{1,{},{}}] = im3[{3,{},{}}]
    im4[{3,{},{}}] = im3[{1,{},{}}]
    im3 = nil
  -- subtract  mean
    for c = 1, 3 do
        im4[c] = im4[c]:sub(1,imheight,1,imwidth) - mean_pix[c]
    end
    return im4
end

function preprocess_batch(im, mean_pix,fixedwid,fixedimh)
    -- rescale the image
    imwidth = fixedwid 
    imheight= fixedimh
    local im3 = image.scale(im:float(),imwidth,imheight,'bilinear')*255
    -- RGB2BGR
    local im4 = im3:clone()
    im4[{1,{},{}}] = im3[{3,{},{}}]
    im4[{3,{},{}}] = im3[{1,{},{}}]
    -- subtract  mean
    for c = 1, 3 do
        im4[c] = im4[c]:sub(1,imheight,1,imwidth) - mean_pix[c]
    end
    return im4
end

function preprocess_gt_deconv(im)
   imheight = fixedimh
   imwidth = fixedwid
    if im:size()[1] > im:size()[2] then
        imwidth = fixedimh
        imheight = fixedwid
    end
    local im3 = image.scale(im:float(),imwidth,imheight,'simple')
    return im3
end

function preprocess_gt_batch(im,fixedwid,fixedimh)
   -- rescale the image
    imwidth = fixedwid/4
    imheight = fixedimh/4
    local im3 = image.scale(im:float(),imwidth,imheight,'simple')
    return im3
 end

function Bilinearkernel(filtersize, noutchannels,ninchannels)
    stridesize = torch.floor((filtersize + 1)/2)
    if stridesize % 2 == 1 then
        center = stridesize - 1
    else
        center = stridesize - 0.5
    end

    og = torch.Tensor(filtersize)

    for i =1,filtersize do
        og[i]=i-1
    end   

    local temp = torch.Tensor(filtersize):fill(1) - (torch.abs(og - torch.Tensor(filtersize):fill(center))/ stridesize)
    local temp = torch.ger(temp,temp) -- outer product
    output =  torch.repeatTensor(temp,ninchannels,noutchannels,1,1)

    for i = 1,ninchannels do
        for j = 1,noutchannels do
            if i~=j then
                output[{{i},{j},{},{}}]:fill(0)
            end
        end
    end
    return output
end
