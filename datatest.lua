require 'image'
matio = require 'matio'

coco_dir = "/share/data/vision-greg/coco/train2014/"
coco_voc = "/share/data/vision-greg/coco/gt-voc/"

tag_dir = "/share/data/vision-greg/mlfeatsdata/voc_tags.t7"

--dat = io.open("/share/data/vision-greg/coco/train.txt","r")
dat = io.open("test.txt","r")
io.input(dat)
temp = {}
for i = 1,66843 do
table.insert(temp, io.read())
end
io.close(dat)

--torch.save(temp, "coco.t7")
img=image.load(coco_dir..string.sub(temp[1],1,-5)..".jpg")
temp_gt=matio.load("/share/data/vision-greg/coco/gt-class/"..temp[1])
gt = temp_gt.groundTruth[1].Segmentation

--print(temp[1])
--torch.save(temp, "coco.t7")

