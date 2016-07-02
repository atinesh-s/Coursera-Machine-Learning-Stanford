function [X_norm, mu, sigma] = featureNormalize(X)
% FEATURENORMALIZE(X) returns a normalized version of X where
% the mean value of each feature is 0 and the standard deviation
% is 1. This is often a good preprocessing step to do when
% working with learning algorithms.

mu = mean(X);
sigma = std(X);

t = ones(length(X), 1);
X_norm = (X - (t * mu)) ./ (t * sigma); % Vectorized

end
