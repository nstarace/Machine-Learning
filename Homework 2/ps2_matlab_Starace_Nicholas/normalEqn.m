function theta = normalEqn(X, y)
    theta = transpose(X) * X \ transpose(X) * y;
end