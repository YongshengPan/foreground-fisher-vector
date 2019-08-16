run('vlfeat\toolbox\vl_setup.m');
run('matconvnet\matlab\vl_setupnn.m');

net = load('imagenet-vgg-m.mat');
net.layers = net.layers(1:13);
net = vl_simplenn_tidy(net);
net = vl_simplenn_move(net, 'gpu');
net.useGpu = true;
scales = 2.^(-3:0.5:3);
boder = 16;
step = 16;
numWords = 128;
numDescrsPerWord = 1000;

annos = bmw10_get_database('bmw10','seed', 0);
% annos = bird6_get_database('bird6','seed', 0);

avgrgb = mean(mean(net.meta.normalization.averageImage));
feats = cell(size(annos.segments.name,1),size(scales,2));
locations = cell(size(annos.segments.name,2),size(scales,2));
norml2    = @(x) bsxfun(@times, x, 1./(sqrt(sum(x.^2,1))+eps));
for idx = 1:length(annos.segments.name)
    im_raw = bsxfun(@minus,single(imread([annos.imageDir, '\', annos.segments.name{idx}])), avgrgb);
    for v = 1:length(scales)
        if sqrt(size(im_raw,1)*size(im_raw,2))*scales(v)>1024, continue; end
        if min(size(im_raw,1),size(im_raw,2))*scales(v)<32, continue; end
        im_resized = imresize(im_raw, scales(v));
        im = gpuArray(im_resized);
        if ndims(im)==2, im_res = cat(3,im,im,im); else, im_res = im; end
        res = vl_simplenn(net, im_res, [], [], ...
                   'conserveMemory', true, 'sync', true);
        feat = permute(gather(res(end).x), [3,1,2,4]);
        local= get_location(size(feat,2),size(feat,3), boder, step, scales(v));
        feat = reshape(feat,size(feat,1),[]);
        local= reshape(local, 2, []);
        feats{idx, v} = norml2(feat);
        locations{idx, v}=local;
    end
end
label = annos.segments.label;
set = annos.segments.set;
dimWords=size(cat(2,feats{1,:}),1);
ENC = cell(size(label,1),1);
ord = zeros(dimWords, numWords*2);
for idx=1:numWords
    ord(:,[idx, idx+numWords]) = idx;
end
[~,ord] = sort(ord(:));

[MEANS, COVARIANCES, PRIORS] = build_gmm(vl_colsubset(cat(2,feats{annos.segments.set==1,:}), numWords*numDescrsPerWord), numWords, 0.0001);
for idx = 1:size(label,2)
    ENC{idx} = vl_fisher(cat(2, feats{idx,:}), MEANS, COVARIANCES, PRIORS, 'Improved');
end
code = cat(2, ENC{:});
svmmodel = linearsvm_smo(code(:,set==1),label(set==1)', 10, annos.classes.name);
save('svmmodel', '-struct', 'svmmodel');
testvals = bsxfun(@minus, code(:, set==3)' * svmmodel.W, svmmodel.rho);
[~,estID] = max(testvals,[],2);
ACC = sum(estID==label(set==3)')/sum(set==3);
disp(['ACC=:', num2str(ACC)]);

function local = get_location(rng_x, rng_y, border, step, scale)
local = zeros(2, rng_x, rng_y);
for idx=1:rng_x
    for idy=1:rng_y
        local(1,idx,:) = (border+(idx-1)*step)/scale;
        local(2,:,idy) = (border+(idy-1)*step)/scale;
    end
end
end


