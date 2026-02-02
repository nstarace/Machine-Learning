function J = computeCost(X, y, theta)
    hypothesis = X * theta;             % Compute Hypothesis = X * Theta
    errorSq = (hypothesis - y) .^ 2;    % Compute absolute squared error 
    sumError = sum(errorSq);            % Compute the summation of errors
    J = sumError / (2 * size(y, 1));    % Normalize Cost
end