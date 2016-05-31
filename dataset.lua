--Load data from disk.

function load_data(filePath)

    local loaded = matio.load(filePath)
    local imlist = loaded.Imlist
    local imlistgt = loaded.Imlistgt
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
    return s, sgt
end
