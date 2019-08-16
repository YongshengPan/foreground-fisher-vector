function svmmodel = linearsvm_smo(X, trainID, costs, classes)

%- Init. -%
n    = size(X,2);
ncls = numel(classes);
svmmodel = struct('C',num2cell(costs), 'rho',[], 'W',[],'alpha',[], 'classes',{classes});
alpha0   = zeros(n, ncls);
KK.K = double(gather(X'*X));
% fprintf('linearSVM with C =   \n');
for i = 1:numel(svmmodel)
	cost = svmmodel(i).C;
	svmparam = sprintf('-c %f -q 1 -i 100000',cost);
% 	fprintf(' %1.1f ',cost);
	coef = zeros(n, ncls);
	rhos = zeros(1, ncls);
	for c = 1:ncls
		if numel(trainID) == n
			% numeric class ID
			y = 2*(trainID == c) - 1;
		else
			% ncls-dimensional binary class ID
			y = trainID(:,c);
		end
		nzidx = (y~=0);
		initalpha = alpha0(:,c);
		[alphas, stats] = smo(y(nzidx), KK.K(nzidx,nzidx), -ones(nnz(nzidx),1),[],initalpha(nzidx),svmparam);

		initalpha(nzidx) = alphas;
		alpha0(:,c)      = initalpha;

		coef(:,c) = double(nzidx).*initalpha.*y;
		rhos(c)   = stats.rho;
%         disp(stats.iter);
	end
	svmmodel(i).W    = X*coef;
	svmmodel(i).rho  = rhos;
    svmmodel(i).alpha  = alpha0;

end
% fprintf('\n');
