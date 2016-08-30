function dsInfo = mapDataSets(database, subset,gtoption,spparam)
% dsInfo = mapDataSets(database,subset,gtoption);
%
% Currently accepted database names, and set names:
%   bsds  : 'train' is BSDS500 train (also BSDS300 train), 'val' is BSDS500 val (also
%            BSDS300 test) and test is BSDS500 test
%   voc   : VOC 2011 (instance level ground truth)
%   voc12 : VOC 2012, with subsets 'train','test','trainval','val',
%         'all' (incl. Berkeley annnotations), 'all-val' (all
%         except val);
%         'sbdtrain' and 'sbdval' are the train/test set of the
%         Semantic Boundary Dataset
%         gtoption specifies which GT to use:
%           inst : instance level (20 classes)
%           class : category level
%           <category> : figure/ground for a given category
%         It is also possible to get only images that
%         include foreground for a given class, e.g.,
%         mapDataSets('voc12','fg-all','areroplane')
%   sbd   : Stanford Background Dataset
%   msrc  : MSRC
%   nyu   : NYU Depth
%
% Also, for IBM medical data:
%   hospitals:
%   Carmel, Ora, China, Benchmark1, ShaareyZedek1, ShaareyZedek2, ShaareyZedek3
%     for all these the subset is ignored (set it to '')
%   Benchmark1 (like hospitals although it's really a sample of
%     images across hospitals)
%   cv5: 5-fold cross val, in which case subset can be 'train<N>' or
%     'test<N>' for N=1,..,5
%   --<DS> e.g. '--ShaareyZedek2' meaning all hospitals except one
%
%   hospitals2 : an extended and cleaned up version of the data set
%
% fields of dsInfo: imnames, segdir, gtdir,spdirs,fdirs
%
% imnames: cell array
% imdir: where the images are
% gtdir : ground truth
% spdirs : .ers, .slic, .ucm point to relevant directories
% fdirs: .gc (geo. context). textons, sift are assumed to be in segdir/<feature>
% textonfile
% extension
%

dsInfo.extension='jpg'; % default

switch database,

  % --------------------------  BSDS
  case {'bsds'}
    switch subset,
      case {'train','small'}
        segdir = '/share/data/vision-greg/BSD/BSR/BSDS500/data/images/train';
        gtdir = '/share/data/vision-greg/BSD/BSR/BSDS500/data/groundTruth/train';
        spdirs.ers='/share/data/vision-greg/BSD/BSR/BSDS500/data/images/train/ers';
        spdirs.ucm='/share/data/vision-greg/BSD/BSR/BSDS500/ucm2/train';
        ioudir=[];
      case 'val',
        segdir='/share/data/vision-greg/BSD/BSR/BSDS500/data/images/val';
        gtdir = '/share/data/vision-greg/BSD/BSR/BSDS500/data/groundTruth/val';
        spdirs.ers='/share/data/vision-greg/BSD/BSR/BSDS500/data/images/val/ers';
        spdirs.ucm='/share/data/vision-greg/BSD/BSR/BSDS500/ucm2/val';
        ioudir=[];
      case 'test'
        segdir='/share/data/vision-greg/BSD/BSR/BSDS500/data/images/test';
        gtdir = '/share/data/vision-greg/BSD/BSR/BSDS500/data/groundTruth/test';
        spdirs.ers='/share/data/vision-greg/BSD/BSR/BSDS500/data/images/test/ers';
        spdirs.ucm='/share/data/vision-greg/BSD/BSR/BSDS500/ucm2/test';
        ioudir=[];
      otherwise,
        error('Unknown Dataset of %s database\n',database);
    end
    spdirs.slic=fullfile(segdir,'SLIC');
    fdirs.gc=fullfile(segdir,'pnm','results');

    % Read in the image file names
    im_files = rdir(fullfile(segdir,'*.jpg'));
    % Store image names in cell array 'imnames'
    imnames = cell(1,length(im_files));
    for i = 1:length(im_files)
        [~,tmp_imname,~] = fileparts(im_files(i).name);
        imnames{i} = tmp_imname;
    end
    if (strcmp(subset,'small'))
      imnames=imnames(1:50);
    end

  % ------------------------------ VOC12
  case 'voc12'
    [~,sysid]=system('echo `whoami;hostname -s`');
    [~,hostname]=system('hostname');
    if (strcmp(hostname(1:4),'greg')) % Greg's laptop
      prefix='v:/';
    else
      prefix='/share/data/vision-greg/';
    end
    imgdir =fullfile(prefix,'Pascal/VOCdevkit/VOC2012/ImageSets/Segmentation');
    clsdir =fullfile(prefix,'Pascal/VOCdevkit/VOC2012/ImageSets/Main');

    stringtolookfor={'pyadolla fourier','pyadolla fermat'};
    loc=find(ismember(stringtolookfor,cellstr(sysid)));
    if ~isempty(loc) && loc==1,
      segdir = '/scratch/voc12/JPEGImages/';
    elseif ~isempty(loc) && loc==2,
      segdir = '/mnt/ssd/voc12/JPEGImages/';
    else
      segdir = fullfile(prefix,'Pascal/VOCdevkit/VOC2012/JPEGImages/');
    end

    % set pointer to GT; always use 'all' (the names according to
    % subset will select the relevant files)
    if (strcmp(subset,'test'))
      gtdir=[];
      ioudir=[];
    else
      gtdir=fullfile(prefix,'Pascal/VOCdevkit/VOC2012/groundTruth',gtoption,'all');
      stringtolookfor={'pyadolla fourier','pyadolla fermat'};
      loc=find(ismember(stringtolookfor,cellstr(sysid)));
      if ~isempty(loc) && loc==1,
        ioudir=fullfile(['/scratch/voc12/ious'],gtoption,'all');
      elseif ~isempty(loc) && loc==2,
        ioudir=fullfile(['/mnt/ssd/voc12/ious'],gtoption,'all');
      else
        ioudir=fullfile(['/share/data/vision-greg/Pascal/VOCdevkit/VOC2012/ious'],gtoption,'all');
      end
    end

    % read in file names
    switch subset,
      case {'train','val','trainval','test','toy','sbdtrain','sbdval','sbdtune'},
        imnames = textread(fullfile(imgdir,[subset '.txt']),'%s');
      case 'all' % include berkeley segmentations
        imnames = textread(fullfile(imgdir,'all_gt_segm.txt'),'%s');
      case 'all_test' % include berkeley segmentations and test images
        imnames = textread(fullfile(imgdir,'all_gt_segm_plus_test.txt'),'%s');
      case 'all-val', % include berkeley segmentations, exclude val
        imnames = textread(fullfile(imgdir,'all_gt_segm_minus_val.txt'),'%s');
      case {'cls-train','cls-val','cls-trainval','cls-test'},
        cls_subset = subset((length('cls-')+1):end);
        imnames = textread(fullfile(clsdir,[cls_subset '.txt']),'%s');
      case 'fg-all',
        imnames = textread(fullfile(gtdir,'fgnames.txt'),'%s');
      case 'fg-all-val',
        imnames = textread(fullfile(gtdir,'fgnames.txt'),'%s');
        valnames = textread(fullfile(imgdir,'val.txt'),'%s');
        for v=1:length(valnames)
          ind=find(strncmp(valnames{v},imnames,length(valnames{v})));
          imnames(ind)=[];
        end
      case 'fg-val',
        imnames = textread(fullfile(gtdir,'fgnames.txt'),'%s');
        valnames = textread(fullfile(imgdir,'val.txt'),'%s');
        ind=[];
        for v=1:length(valnames)
          ind=[ind;find(strncmp(valnames{v},imnames,length(valnames{v})))];
        end
        imnames=imnames(ind);
      otherwise,
        error('Unknown subsset %s of %s database\n',subset,database);
    end

    %spdirs.ucm=fullfile(prefix,'Pascal/VOCdevkit/VOC2012/ucm2');
    spdirs.ucm=fullfile(prefix,'Pascal/mcg/voc-ucm');
    spdirs.slic=fullfile(segdir,'SLIC');
    spdirs.ers=fullfile(prefix,'Pascal/VOCdevkit/VOC2012/ers');
    fdirs.gc=fullfile(prefix,'Pascal/VOCdevkit/VOC2012/pnm/results');
    %dsInfo.textonfile='/share/data/vision-greg/Pascal/VOCdevkit/VOC2012/JPEGImages/textons64/textonscentroids';
    dsInfo.textonfile='textons64/textonscentroids.mat';
    dsInfo.extension='jpg';


  % -------------------------------- Stanford Background Dataset
  case 'sbd',
    imgdir ='/share/data/vision-greg/SBD/ImageSets/Segmentation';
    spdirs.slic='/share/data/vision-greg/SBD/images/SLIC';
    spdirs.ers='/share/data/vision-greg/SBD/ers';
    spdirs.ucm='/share/data/vision-greg/SBD/ucm2';
    fdirs.gc='/share/data/vision-greg/SBD/pnm/results_all';

    imnames={};
    switch subset,
      case 'all-val',
        im_files = fopen([imgdir '/train.txt']);
        gtdir = fullfile('/share/data/vision-greg/SBD/groundTruth/',gtoption);
        ioudir=[];
      case 'val',
        im_files = fopen([imgdir '/val.txt']);
        gtdir = fullfile('/share/data/vision-greg/SBD/groundTruth',gtoption);
        ioudir=[];
      case 'test',
        im_files = fopen([imgdir '/test.txt']);
        gtdir = fullfile('/share/data/vision-greg/SBD/groundTruth',gtoption);
        ioudir=[];
    case 'train+val',
        im_files = fopen([imgdir '/train+val.txt']);
        gtdir = fullfile('/share/data/vision-greg/SBD/groundTruth',gtoption);
        ioudir=[];
      case 'all',
        image_files = rdir('/share/data/vision-greg/SBD/images/*.jpg');
        imnames = cell(1,length(image_files));
        for i = 1:length(image_files)
          [~,name,~] = fileparts(image_files(i).name);
          imnames{i} = name;
        end
        gtdir = fullfile('/share/data/vision-greg/SBD/groundTruth',gtoption);
        ioudir=[];
      otherwise,
        error('Unknown Dataset of %s database\n',database);
    end

    if (isempty(imnames))
      imnames = cell(1,1);
      i = 0;
      while ~feof(im_files)
        i = i + 1;
        imnames{i} = fgetl(im_files);
      end
    end
     dsInfo.textonfile='textons64/textonscentroids.mat';
    segdir = '/share/data/vision-greg/SBD/images';

  % ----------------------------- MSRC
  case 'msrc',
    imgdir = '/share/data/vision-greg/MSRC';

    spdirs.slic='/share/data/vision-greg/MSRC/images/SLIC';
    spdirs.ers='/share/data/vision-greg/MSRC/ers';
    spdirs.ucm='/share/data/vision-greg/MSRC/ucm2';
    fdirs.gc='/share/data/vision-greg/MSRC/pnm/results';

    imnames={};
    switch subset,
      case 'train',
        im_files = fopen([imgdir '/train.txt']);
        gtdir = '/share/data/vision-greg/MSRC/groundTruth/21class';
        ioudir=[];
      case 'val',
        im_files = fopen([imgdir '/val.txt']);
        gtdir = '/share/data/vision-greg/MSRC/groundTruth/21class';
        ioudir=[];
      case 'test',
        im_files = fopen([imgdir '/test.txt']);
        gtdir = '/share/data/vision-greg/MSRC/groundTruth/21class';
        ioudir=[];
      case 'test_class',
        im_files = fopen([imgdir '/test.txt']);
        gtdir = '/share/data/vision-greg/MSRC/groundTruth/class';
        ioudir=[];
      case 'train_class',
        im_files = fopen([imgdir '/train.txt']);
        gtdir = '/share/data/vision-greg/MSRC/groundTruth/class';
        ioudir=[];
      case 'all',
        image_files = rdir('/share/data/vision-greg/MSRC/images/*.bmp');
        imnames = cell(1,length(image_files));
        for i = 1:length(image_files)
          [~,name,~] = fileparts(image_files(i).name);
          imnames{i} = name;
        end
        gtdir = '/share/data/vision-greg/MSRC/groundTruth';
        ioudir=[];
      otherwise,
        error('Unknown Dataset of %s database\n',database);
    end

    if (isempty(imnames))
      imnames = cell(1,1);
      i = 0;
      while ~feof(im_files)
        i = i + 1;
        imnames{i} = fgetl(im_files);
      end
    end

    segdir ='/share/data/vision-greg/MSRC/images';

  % ----------------------------- NYU Depth (only 2d labeled images)
  case 'nyu',
    segdir ='/share/data/vision-greg/NYU_depth/images';
    gtdir = '/share/data/vision-greg/NYU_depth/groundTruth';
    spdirs.ers='/share/data/vision-greg/NYU_depth/ers';
    spdirs.ucm='/share/data/vision-greg/NYU_depth/ucm2';
    spdirs.slic='/share/data/vision-greg/NYU_depth/images/SLIC';
    fdirs.gc='/share/data/vision-greg/NYU_depth/pnm/results';
    ioudir=[];
    if strcmp(subset,'all')
        f = dir(sprintf('%s/*.jpg',segdir));
        imnames = cell(1,length(f));
        for i = 1:length(f)
            [~,f_name,~] = fileparts(f(i).name);
            imnames{i} = f_name;
        end
    else
        error('Unknown Dataset of %s database\n',database);
    end

  % medical data from IBM

  case {'Carmel','Ora','China','Benchmark1','ShaareyZedek1', ...
        'ShaareyZedek2','ShaareyZedek3'},
    if (ispc)
      [~,hostname]=system('hostname');
      if (strcmp(hostname(1:4),'greg')) % Greg's laptop
        segdir=fullfile('c:/greg/consulting/ibm/data/breast-us',database);
      else % IBM issues laptop
        segdir=fullfile('d:/medical/breast-us',database);
      end
    else % linux
      [~,hostname]=system('hostname');
      if (strcmp(hostname(1:min(10,length(hostname))),'lnx-grandc'))
        segdir=fullfile('/home/greg/data/medical/breast-us',database);
      else % TTIC filesystem
        segdir=fullfile('/share/data/vision-greg/greg/ibm/breast-us',database);
      end
    end
    % get im files, that are assumed to be im BMP format
    im_files = rdir(fullfile(segdir,'*.bmp'));
    imnames = cell(1,length(im_files));
    for i = 1:length(im_files)
        [~,tmp_imname,~] = fileparts(im_files(i).name);
        imnames{i} = tmp_imname;
    end
    spdirs.slic = fullfile(segdir,'slic');
    spdirs.ers = [];
    spdirs.ucm = [];
    fdirs.gc=[];
    gtdir=fullfile(segdir,gtoption);
    ioudir=[];

  case {'hospitals','hospitals2','benchmark'}
    if (ispc)
      [~,hostname]=system('hostname');
      if (strcmp(hostname(1:4),'greg')) % Greg's laptop
        %segdir=fullfile('c:/greg/consulting/ibm/data/breast-us',database);
        segdir=fullfile('v:/greg/ibm/breast-us',database);
      else % IBM issued laptop
        error('Pavel must set up path');
      end
    else % linux
      [~,hostname]=system('hostname');
      if (strcmp(hostname(1:min(10,length(hostname))),'lnx-grandc'))
        % IBM server
        segdir=fullfile('/data/data/medical/breast-us/',database);
      else % TTIC filesystem
        segdir=fullfile('/share/data/vision-greg/greg/ibm/breast-us',database);
      end
    end

    switch subset,
      case {'train1','test1','train2','test2','train3','test3','train4','test4','train5','test5'},
        load(fullfile(segdir,[subset '_names.mat'])); % this will
                                                      % load imnames
        numset=str2num(subset(end));
        dsInfo.textonfile=sprintf('train%d_texton%%d.mat',numset);
      case 'all',
        load(fullfile(segdir,'train1_names.mat'));
        imnamestrain=imnames;
        load(fullfile(segdir,'test1_names.mat'));
        imnames=[imnamestrain imnames];
        dsInfo.textonfile=[];
      otherwise,
        error('unknown subset');
    end
    spdirs.slic = fullfile(segdir,'slic');
    spdirs.ers = [];
    spdirs.ucm = [];
    fdirs.gc=[];
    gtdir=fullfile(segdir,gtoption);
    dsInfo.extension='bmp';
    ioudir=[];
  case 'mammo',
    if (ispc)
      [~,hostname]=system('hostname');
      if (strcmp(hostname(1:4),'greg')) % Greg's laptop
        segdir='\\data.ttic.edu\vision-greg\greg\ibm\mammo\images';
      else % IBM issued laptop
        error('Pavel must set path');
      end
    else % linux
      [~,hostname]=system('hostname');
      if (strcmp(hostname(1:min(10,length(hostname))),'lnx-grandc'))
        % IBM server
        error('Pavel must set up path');
      else % TTIC filesystem
        segdir='/share/data/vision-greg/greg/ibm/mammo/images';
      end
    end
    load(fullfile(segdir,sprintf('%s_names.mat',subset)));
    spdirs.slic = fullfile(segdir,'slic');
    spdirs.ers = [];
    spdirs.ucm = [];
    fdirs.gc=[];
    gtdir=fullfile(segdir,'gt');
    dsInfo.extension='bmp';
    ioudir=[];

  case 'horses',
    if (ispc)
      segdir='v:/weizmann-horses/images';
    else % linux
      [~,hostname]=system('hostname');
      if (strcmp(hostname(1:min(10,length(hostname))),'lnx-grandc'))
        % IBM server
        error('Pavel must set up path');
      else % TTIC filesystem
        datadir='/share/data/vision-greg/weizmann-horses';
        segdir=fullfile(datadir,'images');
      end
    end
    % load imnames for train or test
    switch subset,
      case {'train','test','cv1_train','cv2_train','cv3_train','cv4_train','cv5_train','cv1_test','cv2_test','cv3_test','cv4_test','cv5_test'},
        load(fullfile(segdir,sprintf('%s_names.mat',subset)));
      case 'all',
        load(fullfile(segdir,'train_names.mat'));
        imnamestrain=imnames;
        load(fullfile(segdir,'test_names.mat'));
        imnames=[imnamestrain imnames];
    end
    spdirs.slic = fullfile(segdir,'slic');
    spdirs.ers = [];
    spdirs.ucm = [];
    fdirs.gc=[];
    gtdir=fullfile(segdir,'gt');
    dsInfo.extension='bmp';
    dsInfo.textonfile='texton%d.mat';
    ioudir=[];
  case {'graz-bikes','graz-cars','graz-people'},
    oclass=database(6:end);
    if (ispc)
      segdir=['v:/graz02/data/sets/graz/' oclass '/images'];
    else % linux
      [~,hostname]=system('hostname');
      if (strcmp(hostname(1:min(10,length(hostname))),'lnx-grandc'))
        % IBM server
        error('Pavel must set up path');
      else % TTIC filesystem
        datadir=['/share/data/vision-greg/graz02/data/sets/graz/' oclass];
        segdir=fullfile(datadir,'images');
      end
    end
    % load imnames for train or test
    switch subset,
      case {'train','test','cv1_train','cv2_train','cv3_train','cv4_train','cv5_train','cv1_test','cv2_test','cv3_test','cv4_test','cv5_test'},
        load(fullfile(segdir,sprintf('%s_names.mat',subset)));
      case 'all',
        load(fullfile(segdir,'train_names.mat'));
        imnamestrain=imnames;
        load(fullfile(segdir,'test_names.mat'));
        imnames=[imnamestrain imnames];
    end
    spdirs.slic = fullfile(segdir,'slic');
    spdirs.ers = [];
    spdirs.ucm = [];
    fdirs.gc=[];
    gtdir=fullfile(segdir,'gt');
    dsInfo.extension='bmp';
    dsInfo.textonfile='texton%d.mat';
    ioudir=[];


  case 'caltech256',

    setdir='/share/project/similarity/Caltech256';
    catnames=dir(setdir);
    catnames={catnames(3:end-1).name};

    switch subset,
      case 'all',
        flist=dir(fullfile(setdir,'flat-images',[subset '*.jpg']));

      case 'train',
        %TODO: save fixed train/test partition in flat-images
      case 'test',

      otherwise, % assume category 'nnn'
        catidx=find(strncmp(subset,catnames,length(subset)));
        if isempty(catidx),
          error(['unknown subset ' subset ' of Caltech256']);
        end
        flist=dir(fullfile(setdir,'flat-images',[subset '*.jpg']));

    end
    for f=1:length(flist)
      imnames{f}=flist(f).name(1:end-4);
    end

    segdir=fullfile(setdir,'flat-images');
    dsInfo.extension='jpg';
    spdirs=[];
    fdirs=[];
    gtdir=[];

  
  case 'coco',
    % gt options available: 'class','instance','voc'
    
    if (strcmp(gtoption,'instance'))
      error('instance level GT not available yet');
    end
    
    prefix='/share/data/vision-greg/coco/';
    switch subset,
      case {'train','val','trainval','train-voc','val-voc','trainval-voc'},
        imnames=textread(fullfile(prefix,[subset '.txt']),'%s');
        segdir = fullfile(prefix,'trainval2014');
        gtdir = fullfile(prefix,['gt-',gtoption]);
      case 'test',
        imnames=textread(fullfile(prefix,[subset '.txt']),'%s');
        segdir = fullfile(prefix,'test2014');
        gtdir=[];
    end
    
    spdirs.ers=[];
    spdirs.ucm=[];
    spdirs.slic=fullfile(prefix,'slic');
    dsInfo.extension='jpg';
    ioudir=[];
    fdirs=[];
    
    
    
  otherwise,
    error('unknown data set name');
end

dsInfo.dataset = database;
dsInfo.subset = subset;
gtInfo.gtoption = gtoption;
dsInfo.imnames=imnames;
dsInfo.imdir=segdir;
dsInfo.spdirs=spdirs;
dsInfo.fdirs=fdirs;
dsInfo.gtdir=gtdir;
dsInfo.ioudir=ioudir;

if (exist('spparam','var'))
  dsInfo.spparam=spparam;
end

% "methods"
% note: this is problematic since if we change imnames later (e.g.,
% remove part of the files) the binding of these function is
% already set, and wrong images will be loaded!
dsInfo.loadImg=@(i)(imread(fullfile(dsInfo.imdir, ...
                                    [dsInfo.imnames{i},'.',dsInfo.extension])));
dsInfo.loadGT=@(i)(load(fullfile(dsInfo.gtdir,[dsInfo.imnames{i},'.mat'])));

dsInfo.load1GT=@(i)(feval(@(x)(x.groundTruth{1}.Segmentation),load(fullfile(dsInfo.gtdir,[dsInfo.imnames{i},'.mat']))));


dsInfo.loadSuppix=@(i)(cell2mat(struct2cell(load(fullfile(dsInfo.spdirs.(dsInfo.spparam.method),...
                                     nameTemplate(dsInfo.imnames{i},dsInfo.spparam,'',dsInfo.spparam.method)),'suppix'))));

dsInfo.loadSLIC=@(i,k,m)(loadSLIC(dsInfo.spdirs.slic,dsInfo.imnames{i},k,m));

dsInfo.loadNB=@(i,k,m)(loadNB(fullfile(dsInfo.spdirs.slic,nameTemplate(dsInfo.imnames{i},struct('k',k,'m',m,'method','slic','symmetric',true),[],'nb'))));

getucm=@(u)(u.ucm2(3:2:end,3:2:end));

dsInfo.loadUCM=@(i)(getucm(load(fullfile(dsInfo.spdirs.ucm, ...
                                         [dsInfo.imnames{i} '.mat']))));

