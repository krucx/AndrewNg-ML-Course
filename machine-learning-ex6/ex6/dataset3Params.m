function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
val=[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
k=size(val,1);
model = svmTrain(X, y, val(1), @(x1, x2) gaussianKernel(x1, x2, val(1))); 
predictions = svmPredict(model, Xval);
best_error = mean(double(predictions ~= yval));
best_c = val(1);
best_s = val(1);
% You need to return the following variables correctly.

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
for i=1:k
  for j=1:k
    model= svmTrain(X, y, val(i), @(x1, x2) gaussianKernel(x1, x2, val(j))); 
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
  if(error<best_error)
    best_c=val(i);
    best_s=val(j);
    best_error=error;
    endif
  endfor
endfor

C = best_c;
sigma = best_s;





% =========================================================================

end
