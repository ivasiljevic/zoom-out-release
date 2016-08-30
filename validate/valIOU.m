

load /share/data/vision-greg/Pascal/VOCdevkit/VOC2012/JPEGImages/labeling/val/lossesandlabels_slic_k600_m15.mat
addpath(genpath(pwd))

dataset='voc12';
subset='val';
gtoption='class';
systemname='system_local'; 
initspparams.method='slic';
initspparams.symmetric=1;
initspparams.m=15;
initspparams.k=600; 
indices='';% means compute for all images
dsInfo=mapDataSets(dataset,subset,gtoption);
numtrimages=length(dsInfo.imnames);


filepath = 'prediction/pred_%i.mat'
%-- when we have pixel maps
for i=1:length(dsInfo.imnames)
  load(sprintf(filepath,i)); 
  x(x==21)=0;
  seg{i}=x;
  if (mod(i,100)==0) fprintf('%d\n',i); end
end



for i=1:length(dsInfo.imnames)
  G=dsInfo.loadGT(i);
  gt{i}=G.groundTruth{1}.Segmentation;

end 


addpath('/share/data/vision-greg/Pascal/VOCdevkit/VOCcode');


run /share/data/vision-greg/Pascal/VOCdevkit/VOCcode/VOCinit.m

[accuracies,avacc,conf,rawcounts] = VOCevalseg_frommem(VOCopts,'', seg, gt);
