--filePath = '/share/data/vision-greg/mlfeatsdata/unifiedsegnet/Torch/voc12-rand-all-val_GT.mat'
--filePath = "/share/data/vision-greg/mlfeatsdata/CV_Course/voc12-val_GT.mat" end

function load_data(filePath)

    loaded = matio.load(filePath)  --- you don't need to transpose if you use matio
    imlist = loaded.Imlist
    imlistgt = loaded.Imlistgt
    t = {}
    s = {}
    i = 0
    for i = 1, imlist:size()[1] do
        for j = 1, imlist:size()[2]  do
        t[j] = string.char(imlist[{i,j}])
        end
        s[i]=table.concat(t);
    end

    t = {}
    sgt = {}
    i = 0
    for i = 1, imlistgt:size()[1] do
        for j = 1, imlistgt:size()[2]  do
        t[j] = string.char(imlistgt[{i,j}])
        end
        sgt[i]=table.concat(t);
    end

    imlist = s
    imlistgt = sgt
    numimages = #imlist
end
