--Script for loading and training on COCO images.

require 'image'
matio = require 'matio'

COCO_DIR = "/share/data/vision-greg/coco/train2014/"
COCO_VOC = "/share/data/vision-greg/coco/gt-voc/" --voc ground truth for COCO
TAG_DIR = "/share/data/vision-greg/mlfeatsdata/voc_tags.t7"

dat = io.open("coco_mat.txt","r")
io.input(dat)
im_path = {}
for i = 1,66843 do
table.insert(im_path, io.read())
end
io.close(dat)

-- Copy lines below into training script to load (1) images and (2) GT, here first img/GT.
img=image.load(COCO_DIR..string.sub(im_path[1],1,-5)..".jpg")
temp_gt=matio.load("/share/data/vision-greg/coco/gt-class/"..im_path[1])
gt = temp_gt.groundTruth[1].Segmentation
