function [MEANS, COVARIANCES, PRIORS] = build_gmm( descrs, numWords, CovarianceBound)
%BUILD_GMM Summary of this function goes here
%   Detailed explanation goes here

[MEANS, COVARIANCES, PRIORS] = vl_gmm(descrs, numWords,'Initialization','kmeans','CovarianceBound',CovarianceBound,'NumRepetitions',1);
if ~exist('order','var'), order = 'descend'; end
[PRIORS,r]=sort(PRIORS, order);

MEANS=MEANS(:,r);
COVARIANCES=COVARIANCES(:,r);

% PRIORS=PRIORS(1:numWords);
% MEANS=MEANS(:,1:numWords);
% COVARIANCES=COVARIANCES(:,1:numWords);

end

