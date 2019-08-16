function imdb = bird6_get_database(dataDir, varargin)
opts.seed = 0 ;
opts = vl_argparse(opts, varargin) ;

rng(opts.seed, 'twister');

imdb.imageDir = fullfile(dataDir, 'image') ;

cats = dir(imdb.imageDir) ;
cats = cats([cats.isdir] & ~ismember({cats.name}, {'.','..'})) ;
imdb.classes.name = {cats.name};
imdb.images.id = [] ;

for c=1:numel(cats)
  ims = dir(fullfile(imdb.imageDir, imdb.classes.name{c}, '*.jpg'));
  imdb.images.name{c} = cellfun(@(S) fullfile(imdb.classes.name{c}, S), ...
    {ims(randperm(numel(ims))).name}, 'Uniform', 0);
  imdb.images.label{c} = c * ones(1,numel(ims)) ;
  if numel(ims) < 50, error('ops') ; end
  sets = [1 * ones(1,50), 3 * ones(1,numel(ims)-50)] ;
  imdb.images.set{c} = sets(randperm(numel(ims))) ;
end
imdb.images.name = horzcat(imdb.images.name{:}) ;
imdb.images.label = horzcat(imdb.images.label{:}) ;
imdb.images.set = horzcat(imdb.images.set{:}) ;

imdb.images.id = 1:numel(imdb.images.name) ;

imdb.segments = imdb.images;
imdb.segments.imageId = imdb.images.id ;
imdb.segments.mask = strrep(imdb.images.name, 'image', 'mask') ;

imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
imdb.segments.difficult = false(1, numel(imdb.segments.id)) ;

