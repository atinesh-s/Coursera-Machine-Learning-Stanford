function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

results = eye(64,3);
errorRow = 0;

for C_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
    for sigma_test = [0.01 0.03 0.1 0.3 1, 3, 10 30]
        errorRow = errorRow + 1;
        model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));

        results(errorRow,:) = [C_test, sigma_test, prediction_error];     
    end
end

sorted_results = sortrows(results, 3); % sort matrix by column #3, the error, ascending

C = sorted_results(1,1);
sigma = sorted_results(1,2);

% =========================================================================

end
