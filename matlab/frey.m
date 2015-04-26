% problem parameters
T = 10; W = T; D = T; F = eye(T);

% prepare runs
rand('seed', 421);
n_runs = 100;

add_decorr = 0;
verbose = 0;

n_errors = 0;
for run=1:n_runs
    noise_decay = 5;

    % prepare matrices
    Phi = rand(T); Phi = Phi ./ repmat(sum(Phi), W, 1);
    %Phi = ones(T) ./ T;
    Theta = rand(T); Theta = Theta ./ repmat(sum(Theta), T, 1);
    %Theta = ones(T) ./ T;

    % run algorithm (Frey's code)
    is_converged = 0;
    n_iterations = 0;
    while !is_converged
        n_iterations += 1;

        % save matrices from previous step
        Phi_prev = Phi;
        Theta_prev = Theta;

        % do EM
        Z = F./ (Phi * Theta); Z(F==0) = 0; % this line is correct but really slow
        Phi_tmp = Phi .* (Z * Theta');
        if add_decorr
            % ramdomized decorrelator
            tau = 0.5;
            decorrelator = Phi_tmp .* (Phi_tmp * (F == 0));
            noise = (rand(T)-0.5) .* (decorrelator > 0) / noise_decay;
            noise_decay += 5;
            
            Phi_tmp -= tau * decorrelator;
            Phi_tmp(Phi_tmp < 0) = 0;
        end
        Theta_tmp = Theta .* (Phi' * Z);    
        Phi = Phi_tmp ./ repmat(sum(Phi_tmp), W, 1);
        Theta = Theta_tmp ./ repmat(sum(Theta_tmp), T, 1);

        % check if algorithm converged
        if and(Phi_prev == Phi, Theta_prev == Theta)
            is_converged = 1;
        end
    end

    % check if decomposition is correct
    if or(
        Phi * Theta != F, 
        !check_permutation_matrix(Phi),
        !check_permutation_matrix(Theta)
       )

        n_errors += 1;
        if verbose
            run
            Phi
            Theta
            Phi * Theta
        end
    end
end
printf('Correct decomposition in %.2f %% cases\n', 100 * (1 - n_errors / n_runs));
