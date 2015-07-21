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
value_vect = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_vect = ones(length(value_vect).^2, 1)*-1;
for c_counter = 1:length(value_vect)
    for s_counter = 1:length(value_vect)
        model= svmTrain(X, y, value_vect(c_counter), ...
            @(x1, x2) gaussianKernel(x1, x2, value_vect(s_counter)));
        predictions = svmPredict(model, Xval);
        error_vect((c_counter-1)*length(value_vect) + s_counter) = ...
            mean(double(predictions ~= yval));
    end
end

min_index = find(error_vect == min(error_vect), 1, 'first');

[sigma_ind, C_ind] = ind2sub([length(value_vect) length(value_vect)],...
    min_index);

C = value_vect(C_ind); sigma = value_vect(sigma_ind);
% =========================================================================

end
