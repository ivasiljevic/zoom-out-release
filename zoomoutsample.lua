
function sparse_zoomout_features(zoomout_model, dataset, mean, std,coordinate)
	--[[
	Extract sparse set of features from zoomout_model.
	zoomout_model:
	dataset: 
	mean/sd: Optional mean/variance normalization to improve features.
	coordinate: Specify optional coordinate tensors.
	--]]

	std = std or torch.Tensor(8320)
	mean = mean or torch.Tensor(8320)
	--std = torch.load("samp_std.t7"):squeeze()
	--mean = torch.load("samp_mean.t7"):squeeze()

	for i=1,std:size()[1] do
		if std[i] ==0 then
		std[i] = 1
		end
	end

	num_images = dataset:size()

	local samplabels = torch.DoubleTensor(num_images*600):zero():float()
	local sampfeats = torch.DoubleTensor(num_images*600,8322):zero():float()

	for k=1,num_images do
		collectgarbage()
		local counter = 0
		local flagcl = torch.Tensor(21):fill(0);

		local im = image.load(s[k])
		local loaded = matio.load(sgt[k]) 
		local im_proc_temp = preprocess(im,mean_pix)

		local im_proc = torch.Tensor(1,3,im_proc_temp:size()[2],im_proc_temp:size()[3])
		im_proc[{{1},{},{},{}}] = im_proc_temp

		local datasetlabels =  preprocess_gt(loaded.GT)
		local randi = torch.randperm(datasetlabels:size()[1])
		local randj = torch.randperm(datasetlabels:size()[2])

		local concatfeats = zoomout_model:forward(im_proc:cuda())
		local repl = nn.Replicate(concatfeats[1]:size()[4],1):cuda()
		local temp = repl:forward(concatfeats[2]);
		local repl = nn.Replicate(concatfeats[1]:size()[3],2):cuda()
		local globfeats = repl:forward(temp):transpose(3,1):transpose(2,4);
		Join = nn.JoinTable(2):cuda():forward({concatfeats[1], globfeats});
		globfeats = nil
		concatfeats = nil
		temp = nil
		repl = nil

		for tt = 1,Join:size()[2] do
			Join[{{},{tt},{},{}}]:add(-mean[tt])
			Join[{{},{tt},{},{}}]:div(std[tt])
		end

		if coordinate then 
			local cord_x = coordinate_x(Join:size()[3],Join:size()[4])
			local cord_y = coordinate_y(Join:size()[3],Join:size()[4])
			local x = cord_x:resize(1,1,Join:size()[3],Join:size()[4])
			local y = cord_y:resize(1,1,Join:size()[3],Join:size()[4])
			Join= torch.cat(Join:float(),y,2)
			Join = torch.cat(Join, x,2)
		end
		
		for i = 1, datasetlabels:size()[1] do
			for j = 1, datasetlabels:size()[2] do
				clabel = torch.squeeze(datasetlabels[{{randi[i]},{randj[j]}}])

				if flagcl[clabel] < 100 then
					counter = counter + 1
					sampfeats[{{counter},{}}] = Join[{{1},{},{randi[i]},{randj[j]}}]:squeeze():float()
					samplabels[counter] = clabel
					flagcl[clabel] = flagcl[clabel] +1
				end
			end
		end

	datasetlabels = nil

	xlua.progress(k,num_images)
	end

torch.save("sampling/sample_feats.t7",sampfeats)
end
