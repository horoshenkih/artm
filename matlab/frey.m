% problem parameters
T = 10; W = T; D = T; F = eye(T);

n_runs = 100;
n_errors = 0;

initial_seed = 42;
for run=1:n_runs
    % prepare RNG
    rand('seed', initial_seed + run)

    % prepare matrices
    Phi = rand(T); Phi = Phi ./ repmat(sum(Phi), W, 1);
    Theta = rand(T); Theta = Theta ./ repmat(sum(Theta), T, 1);
    %Theta = ones(T) ./ T;

    % run algorithm (Frey's code)
    for i=1:50
        Z = F ./ (Phi * Theta); Z(F==0) = 0; % this line is correct but really slow
        Phi_tmp = Phi .* (Z * Theta');
        Theta_tmp = Theta .* (Phi' * Z);    
        Phi = Phi_tmp ./ repmat(sum(Phi_tmp), W, 1);
        Theta = Theta_tmp ./ repmat(sum(Theta_tmp), T, 1);
    end

    product = Phi * Theta;

    if or(product != F, 
        !check_permutation_matrix(Phi),
        !check_permutation_matrix(Theta))

        n_errors += 1;
    end
end
printf('Correct decomposition in %.2f %% cases\n', 100 * (1 - n_errors / n_runs));
