function [theta, cost_history] = gradientDescent(X, y, theta_init, alpha, num_iter)
    m = length(y);
    cost_history = zeros(num_iter, 1);                    

    theta = theta_init;
    for i = 1:num_iter
        hypothesis = X * theta;             % Prediction [m x 1]
        err = hypothesis - y;               % Error [m x 1]
        errX = transpose(X) * err;          % [n x m][m x 1] = [n x 1] 
        update = (alpha / m) * errX;        % Error along features, samples used
        theta = theta - update;
        cost_history(i) = computeCost(X, y, theta);
    end
end