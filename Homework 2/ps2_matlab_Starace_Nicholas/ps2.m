D = 22;                                 % Personalization Parameter

% Question 1
% Test Compute Cost
X = [1 0 1; 1 1 1.5; 1 2 4; 1 3 2];     % [4 x 3]
y = [1.5+(D/100); 4; 8.5; 8.5+(D/50)];  % [3 x 1]
theta = [0.5; 2; 1];                    % [3 x 1]
computeCost(X, y, theta)

% Test Gradient Descent   
theta_init = [0; 0; 0];
alpha = 0.01;
num_iter = 100 + (D * 10);
[theta, cost_history] = gradientDescent(X, y, theta_init, alpha, num_iter);
figure (1);
plot(cost_history);
xlabel("Iteration Number");
ylabel("Cost");
xlim([0 320]);
theta

% Test Normal Equation
theta = normalEqn(X, y);
theta

% Question 2
Dataset = readmatrix('input\elec_consump.csv');     % Import X and y
X = Dataset(:, 2:5);                                % Parse X from y
y = Dataset(:, 6);
validRows = all(~isnan(X), 2) & ~isnan(y);          % Remove rows with any NaNs
X = X(validRows, :);
y = y(validRows);
m = length(y);

figure (2);                                         % Plot Feature Space
plot(X(:,1), y, '.');
xlabel('Average Daily Wind Speed [m/s]');
ylabel('Daily Electricity Consumption [kWh]');
title('Electricity vs. Wind Speed');
figure (3);
plot(X(:,2), y, '.');
xlabel('Daily Precipitation [mm]');
ylabel('Daily Electricity Consumption [kWh]');
title('Electricity vs. Precipitation');
figure (4);
plot(X(:,3), y, '.');
xlabel('Daily Max Temperature [Deg Celsius]');
ylabel('Daily Electricity Consumption [kWh]');
title('Electricity vs. Max Temp');
figure (5);
plot(X(:,4), y, '.');
xlabel('Daily Min Temperature [Deg Celsius]');
ylabel('Daily Electricity Consumption [kWh]');
title('Electricity vs. Min Temp');

% Standardization
Means = mean(X, 'omitnan');                     % Omit NaN since can't compute                    
Stdev = std(X, 'omitnan');   
stdX = (X - Means) ./ Stdev;
stdXbias = [ones(m, 1) stdX];

% Seed Random Splt
rng(22);
splitRows = randperm(m);                        % Random Vector of Row indexes
Thresh = floor(0.9 * m);                        % Round to Integer for indexing
X_train = stdXbias(splitRows(1:Thresh), :);
X_test = stdXbias(splitRows(Thresh+1:end), :);
y_train = y(splitRows(1:Thresh), :);
y_test = y(splitRows(Thresh+1:end), :);

% Univariate Regression
theta_init = [0; 0];
alpha = 0.1;
num_iter = 500 + (D * 5);
UniX = X_train(:, 1:2);                         % Wind Speed (feature 1) due to positive correlation
[thetaUni, cost_history] = gradientDescent(UniX, y_train, theta_init, alpha, num_iter);
hyp = UniX * thetaUni;
thetaUni

% Plot Theta Conversion
figure (6);
plot(cost_history);
xlabel("Iteration Number");
ylabel("Cost");
xlim([0 num_iter]);

% Construct Regression Line
xLine = linspace(min(X(:,1)), max(X(:,1)), 100)';       % Extend Domain of Line
XLineBias = [ones(size(xLine)) xLine];                  % Create Bias for Theta Mult
yLine = XLineBias * thetaUni;                              % Get Hypothesis over full domain
figure (7);
plot(X(:,1), y, 'b.', XLineBias(:,2), yLine, 'r');      % Dont plot bias term
xlabel("Wind Speed [m/s]");
ylabel("Daily Electricity Consumption [kWh]");

% Multivariate Regression
theta_init = [0; 0; 0; 0; 0];                           % 5 Features including bias
alpha = 0.1;
num_iter = 750 + (D * 5);
tic;
thetaNormal = normalEqn(X_train, y_train);
elapsedNormal = toc
tic;
[thetaGrad, cost_history] = gradientDescent(X_train, y_train, theta_init, alpha, num_iter);
elapsedGradient = toc

% Plot Theta Conversion
figure (8);
plot(cost_history);
xlabel("Iteration Number");
ylabel("Cost");
xlim([0 num_iter]);

thetaNormal
thetaGrad

% Model Evaluation
CostGrad = computeCost(X_test, y_test, thetaGrad)           % Predict ytest using theta from Gradient Descent
CostNorm = computeCost(X_test, y_test, thetaNormal)         % Predict ytest using theta from Normal Equation
CostUniv = computeCost(X_test(:,1:2), y_test, thetaUni)     % Predict ytest using theta from Univariate Gradient Descent

% Learning Rate Analysis
theta_init = [0; 0; 0; 0; 0];                           % 5 Features including bias

% Slow Convergence
alphaSlow = 0.01
num_iter = 300;
tic;
[thetaSlow, cost_history] = gradientDescent(X_train, y_train, theta_init, alphaSlow, num_iter);
elapsedSlow = toc;
figure (9);
plot(cost_history);
xlabel("Iteration Number");
ylabel("Cost");
title("Slow Convergence");
xlim([0 num_iter+50]);

% Approximate Convergence
alphaSmooth = 0.86
num_iter = 300;
tic;
[thetaApprox, cost_history] = gradientDescent(X_train, y_train, theta_init, alphaSmooth, num_iter);
elapsedApprox = toc;
figure (10);
plot(cost_history);
xlabel("Iteration Number");
ylabel("Cost");
title("Smooth Convergence");
xlim([0 num_iter+50]);

% Oscillating Convergence
alphaOsc = 0.8678
num_iter = 300;
tic;
[thetaOsc, cost_history] = gradientDescent(X_train, y_train, theta_init, alphaOsc, num_iter);
elapsedOsc = toc;
figure (11);
plot(cost_history);
xlabel("Iteration Number");
ylabel("Cost");
title("Oscillating to Convergence");
xlim([0 num_iter + 50]);

% Divergence
alphaDiver = 0.87
num_iter = 300;
tic;
[thetaDiver, cost_history] = gradientDescent(X_train, y_train, theta_init, alphaDiver, num_iter);
elapsedDiver = toc;
figure (12);
plot(cost_history);
xlabel("Iteration Number");
ylabel("Cost");
title("Divergence");
xlim([0 num_iter + 50]);
