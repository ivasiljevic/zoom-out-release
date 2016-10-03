--Validation function.

function validate(model)
    for k=1,#imlist do
        collectgarbage()
        model:evaluate()
        local im = image.load(s[k])
        local im_proc_temp = preprocess(im,mean_pix,fixedimh,fixedimw)
        local im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
        im_proc[{{1},{},{},{}}] = im_proc_temp

        local pred = model:forward(im_proc:cuda())

        gt_temp = nil
        concatfeats = nil

        im_rescale = image.scale(pred[1]:float(),im:size()[3],im:size()[2],"bilinear")
        C,D = torch.max(im_rescale,1)
        gt = D:squeeze():double()
        matio.save('prediction/pred_'..k..'.mat',gt)
        xlua.progress(k,#imlist)
    end
end

