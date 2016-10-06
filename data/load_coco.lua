--Script for loading and training on COCO images.

require 'image'
matio = require 'matio'

COCO_DIR = "/share/data/vision-greg/coco/train2014/"
--COCO_VOC = "/share/data/vision-greg/coco/gt-voc/" --voc ground truth for COCO
--TAG_DIR = "/share/data/vision-greg/mlfeatsdata/voc_tags.t7"

coco_dat = io.open("/home-nfs/ivas/zoom-out-release/data/coco_mat.txt","r")
io.input(coco_dat)
im_path = {}
numimages = 66743
for i = 1,numimages do
table.insert(im_path, io.read())
end
io.close(coco_dat)
-- Copy lines below into training script to load (1) images and (2) GT, here first img/GT.
--img=image.load(COCO_DIR..string.sub(im_path[1],1,-5)..".jpg")
--temp_gt=matio.load("/share/data/vision-greg/coco/gt-class/"..im_path[1])
--gt = temp_gt.groundTruth[1].Segmentation
