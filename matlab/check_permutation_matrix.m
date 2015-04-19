function y = check_permutation_matrix(X)
    n = columns(X);
    % check matrix size
    y = and(
        rows(X) == n,
        sum(sum(X == 0),2) == n*(n-1),
        all(sum(X) == 1),
        all(sum(X,2) == 1)
    );
end
