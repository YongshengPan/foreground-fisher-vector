function imdb = bmw_get_database(dataDir, varargin)
opts.seed = 0 ;
opts = vl_argparse(opts, varargin) ;

rng(opts.seed, 'twister') ;

imdb.imageDir = fullfile(dataDir, 'image') ;

annos = load(fullfile(dataDir,'bmw10_annos.mat'));

imdb.classes.name = num2cell([1:8,10,11]');
imdb.images.name  = {annos.annos.fname}';
imdb.images.label = [annos.annos.class];
imdb.images.set   = ones(1,length(imdb.images.name));
imdb.images.set(annos.train_indices) = 1;
imdb.images.set(annos.test_indices) = 3;
imdb.images.id    = 1:numel(imdb.images.name) ;

imdb.segments = imdb.images ;
imdb.segments.imageId = imdb.images.id ;

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
imdb.segments.difficult = false(1, numel(imdb.segments.id)) ;


