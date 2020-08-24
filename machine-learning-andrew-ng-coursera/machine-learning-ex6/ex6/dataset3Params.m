function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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

steps = 8;
error = 0;
c_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

model= svmTrain(X, y, c_values(1), @(x1, x2) gaussianKernel(x1, x2, sigma_values(1)));
predictions = svmPredict(model, Xval);
pred_error = mean(double(predictions ~= yval));
error = pred_error;

for i = 1:steps
    for j = 1:steps
        model= svmTrain(X, y, c_values(i), @(x1, x2) gaussianKernel(x1, x2, sigma_values(j)));
        predictions = svmPredict(model, Xval);
        pred_error = mean(double(predictions ~= yval));
        %fprintf("pred_error = %f\n", pred_error);
        if (pred_error < error),
            C = c_values(i);
            sigma = sigma_values(j);
            error = pred_error;
        end
    end
end

% =========================================================================

end
