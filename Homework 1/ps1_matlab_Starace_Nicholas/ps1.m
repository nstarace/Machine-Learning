clc
clear
% Question 2
% Part A
D = 22;                                     % Personalization Parameter
Mean = 2.0 + (D / 100);
StdDev = 0.5 + (D / 200);
x = randn(1000000, 1);                      % 1,000,000 x 1 vector of random samples from Gaussian distribution
x = Mean + StdDev .* x;                     % Configure mean and standard deviation

% Part B
Range = [(-D/50) (D/100)];
z = rand(1000000, 1);                       % Generate from uniform distribution
z = Range(1) + (Range(2) - Range(1)).*z;    % Close range

% Part C
figure (1);
histogram(x, 'Normalization', 'pdf');       % Plot Normalized Histogram so all samples sum to 1
xlabel('Intensity');
ylabel('Probability Density');
title('Normalized Histogram of x');
figure (2);
histogram(z, 'Normalization', 'pdf');
xlabel('Intensity');
ylabel('Probability Density');
title('Normalized Histogram of z');

% Part D
tic;
for i = 1:size(x)
    x(i) = x(i) + 2;                        % Iterate over x and add 2 to each element
end
elapsedLoop = toc;                          % Time iteration

tic;
x = x + 2;                                  % Vectorized addition
elapsedVector = toc;                        % Time vectorized

% Part E
k = 0;
for i = 1:size(z)
    if z(i) > 0 && z(i) < 0.8
        k = k + 1;
        y(k) = z(i);                        % Size unknown so increment size
    end
end


% Question 3
% Part A
A = [2 10 8; 3 5 2; 6 4 4];
fprintf("Minimum per Column: %d %d %d\n", min(A, [], 1));  % Min along column dimension
fprintf("Maximum per Row: %d %d %d\n", max(A, [], 2));     % Max along row dimension
fprintf("Global Minimum: %d\n", min(A, [], 'all'));        % Global min
fprintf("Sum per Row: %d %d %d\n", sum(A, 2));             % Sum of each row
fprintf("Global Sum: %d\n", sum(A, 'all'));                % Global sum
fprintf("Square of Each Element:");                        % Square of each element
B = A .^ 2

if B(1,2) ~= A(1,2) ^ 2
    fprintf("B does not equal square of A\n");
else
    fprintf("B is the square of A\n");
end

% Part B
S = [2 5 -2; 2 6 4; 6 8 18];
T = [D; 6; 15];
U = S\T;                                                    % Sx = T

% Question 4 
% Part A
X = zeros(10, 3);
for i = 1:10
    X(i) = i;                                               % Iterate only 10 times
end
X(:,2) = X(:,1) .^ 2;                                       % Vectorized Compute
X(:,3) = X(:,1) .* D;
y = 3 .* X(:,1) + D;                                        % Copy from Existing
X                                                           % Print Data
y

% Part B
split_rows = randperm(10);                                  % Generate random permutations for row indices
X_train = X(split_rows(1:8), :);
X_test = X(split_rows(9:10), :);
y_train = y(split_rows(1:8), :);
y_test = y(split_rows(9:10), :);

X_train
X_test
y_train
y_test

% Part C
rng(37);                                                    % Preseed randomness for reproduction
split_rows = randperm(10);                                  % Generate random permutations for row indices
X_train = X(split_rows(1:8), :);
X_test = X(split_rows(9:10), :);
y_train = y(split_rows(1:8), :);
y_test = y(split_rows(9:10), :);

fprintf("Modfied for Preseeded Randomness\n");
X_train
X_test
y_train
y_test










