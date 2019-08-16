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
max_itr = 2;
numWords = 128;
numDescrsPerWord = 1000;
fg_obj = 0.1;  
% annos = bird6_get_database('bird6','seed', 0);
annos = bmw10_get_database('bmw10','seed', 0);

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
posgmmparams = struct();
for itr = 1 : max_itr
    [MEANS, COVARIANCES, PRIORS] = build_gmm(vl_colsubset(cat(2,feats{annos.segments.set==1,:}), numWords*numDescrsPerWord), numWords, 0.0001);
    for idx = 1:size(label,2)
        ENC{idx} = vl_fisher(cat(2, feats{idx,:}), MEANS, COVARIANCES, PRIORS, 'Improved');
    end
    ifv_code = cat(2, ENC{:});
    [sfv_code, V] = softassumption(ifv_code(ord,:), label, set, dimWords*2, annos.classes.name);
    posgmmparams.MEANS = MEANS;
    posgmmparams.COVARIANCES = COVARIANCES;
    posgmmparams.PRIORS = PRIORS;
    posgmmparams.V = V;
    save(['posgmmparams_epoch',num2str(itr)], '-struct', 'posgmmparams');
    svmmodel = linearsvm_smo(sfv_code(:,set==1),label(set==1)', 10, annos.classes.name);
    save('svmmodel', '-struct', 'svmmodel');
    testvals = bsxfun(@minus, sfv_code(:, set==3)' * svmmodel.W, svmmodel.rho);
    [~,estID] = max(testvals,[],2);
    ACC = sum(estID==label(set==3)')/sum(set==3);
    disp(['ACC=:', num2str(ACC)]);
    
    for idx=1:length(feats)
        if itr == max_itr
            file_name = [annos.imageDir, '\', annos.segments.name{idx}];
            im_raw = im2single(imread(file_name));
            if ndims(im_raw)==2, im_raw = cat(3,im_raw,im_raw,im_raw); end
            im_seg = gpuArray(im_raw-im_raw);
            im_seg_temp = im_seg;
        end
        for idy=1:length(scales)
            if isempty(feats{idx, idy}), continue; end
            prob = V'*posgmm(feats{idx, idy}, MEANS, COVARIANCES, PRIORS');
            feats{idx, idy} = feats{idx, idy}(:,prob>1e-3);
            locations{idx, idy} = locations{idx, idy}(:,prob>1e-3);
            if itr == max_itr
                prob = prob(prob>1e-3);
                im_seg_temp(:)=0;
                for idz=1:size(locations{idx, idy},2)
                    loc = locations{idx, idy}(:,idz)+1;
                    x_min = max(1,loc(1)-8/scales(idy));
                    y_min = max(1,loc(2)-8/scales(idy));
                    x_max = min(size(im_seg,1),loc(1)+8/scales(idy));
                    y_max = min(size(im_seg,2),loc(2)+8/scales(idy));
                    im_seg_temp(round(x_min:x_max),round(y_min:y_max),:) = (prob(idz)).^ fg_obj;
                end
                im_seg = im_seg+im_seg_temp;
            end
        end
        if itr == max_itr
            im_seg = gather(im_seg);
            im_map = (im_seg-min(im_seg(:)))/(max(im_seg(:))-min(im_seg(:)));
            [PATHSTR,NAME,EXT] = fileparts(strrep(file_name, 'image', 'map'));
            if ~exist(PATHSTR, 'dir'), mkdir(PATHSTR); end
            imwrite(im_map, [PATHSTR,'\',NAME,EXT]);
            gausFilter = fspecial('gaussian', [31,31], round(sqrt(size(im_map,1)*size(im_map,2))/32));
            im_seg = imfilter(im_map, gausFilter,'same');
            SE = strel('disk',round(sqrt(size(im_map,1)*size(im_map,2))/16));
            im_seg = imclose(im_seg(:,:,1)>0.5,SE);
            im_seg = imfill(im_seg, 'holes');
            im_seg = imopen(im_seg,SE);
            [PATHSTR,NAME,EXT] = fileparts(strrep(file_name, 'image', 'seg'));
            if ~exist(PATHSTR, 'dir'), mkdir(PATHSTR); end
            imwrite(im_seg, [PATHSTR,'\',NAME,EXT]);
            im_fg = bsxfun(@times, im_raw, im_seg);
            [PATHSTR,NAME,EXT] = fileparts(strrep(file_name, 'image', 'fg'));
            if ~exist(PATHSTR, 'dir'), mkdir(PATHSTR); end
            imwrite(im_fg, [PATHSTR,'\',NAME,EXT]);
            im_mix = cat(1, im_raw, im_map, cat(3,im_seg,im_seg,im_seg), im_fg);
            [PATHSTR,NAME,EXT] = fileparts(strrep(file_name, 'image', 'mix'));
            if ~exist(PATHSTR, 'dir'), mkdir(PATHSTR); end
            imwrite(im_mix, [PATHSTR,'\',NAME,EXT]);
        end
    end
end

function local = get_location(rng_x, rng_y, border, step, scale)
    local = zeros(2, rng_x, rng_y);
    for idx=1:rng_x
        for idy=1:rng_y
            local(1,idx,:) = (border+(idx-1)*step)/scale;
            local(2,:,idy) = (border+(idy-1)*step)/scale;
        end
    end
end

function E = posgmm(X, m, sig2, w)
norml1 = @(x) bsxfun(@times, x, 1./sum(x,1));
invsig2 = 1./sig2;
msig = bsxfun(@times, m, invsig2);
E = bsxfun(@plus, bsxfun(@minus, bsxfun(@minus,msig'*X,0.5*sum(m.*msig,1)'), (0.5*invsig2')*(X.^2)), (log(w)+0.5*sum(log(invsig2),1))');
maxE = max(E,[],1);
E = exp(bsxfun(@minus,E,maxE));
E = norml1(E);
E(E<1e-4) = 0;
E = norml1(E);
end

function [code, V_rcd] = softassumption(code, label, set, dimWords, classes)
if ~ exist('tmpPath', 'var')
    tmpPath = 'tmpfile.mat';
end
trainID = ismember(set, 1) ;
valID   = ismember(set, 2) ;
testID  = ismember(set, 3) ;
if sum(valID)==0
    for idx = unique(label)
        loc = (set==1) & (label==idx);
        set(loc) = [1*ones(1,round(sum(loc)/2)),2*ones(1, sum(loc)-round(sum(loc)/2))];
    end
    trainID = ismember(set, 1) ;
    valID   = ismember(set, 2) ;
end
numWords= size(code,1)/dimWords;
V = ones(numWords, 1);
V_rcd = V;
norml2 = @(x) bsxfun(@times, x, 1./(sqrt(sum(x.^2,1))+eps));
maxiter = 2;
for itr = 1:maxiter
    svmmdl = linearsvm_smo(code(:,trainID),label(trainID)', 10, classes);
    info.W = svmmdl.W;
    info.rho = svmmdl.rho;
    info.label = label(:,valID);
    info.set = set(:, valID);
    info.dimPerWords = dimWords;
    info.numWords = numWords;
    info.psc = zeros(sum(valID),size(info.W,2), numWords);
    info.ssc = zeros(sum(valID), numWords);
    for idx = 1: numWords
        pX = (code(dimWords*(idx)+1-(1:dimWords),valID));
        info.psc(:, :, idx) =gather( pX'* (info.W(dimWords*(idx)+1-(1:dimWords),:)));
        info.ssc(:, idx) = gather(sum(pX.*pX));
    end
    info.V = V;
    save(tmpPath, '-v7.3', '-struct', 'info') ;
    [status,sttout] = system(['python sfv.py ', tmpPath]);
    if status
        error(sttout);
    end
    load(tmpPath,'V');
    V_rcd(V_rcd>0) = V;
    if itr < maxiter
        V = single(V > 0);
    end
    for idx = 1: numWords
        code(dimWords*(idx)+1-(1:dimWords),:) = code(dimWords*(idx)+1-(1:dimWords),:)*V(idx);
    end
    code = code(repmat(V'>0, dimWords, 1), :);
    code = norml2(code);
    numWords= size(code,1)/dimWords;
    V = ones(numWords, 1);
end
end
