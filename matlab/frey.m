% problem parameters
T = 10; W = T; D = T; F = eye(T);

% prepare runs
rand('seed', 42);
n_runs = 100;
n_errors = 0;

noise_decay = 5;

add_f_noise = 0;
add_phi_noise = 0;

add_decorr = 1;
verbose = 0;

for run=1:n_runs
    % prepare matrices
    Phi = rand(T); Phi = Phi ./ repmat(sum(Phi), W, 1);
    %Phi = ones(T) ./ T;
    Theta = rand(T); Theta = Theta ./ repmat(sum(Theta), T, 1);
    %Theta = ones(T) ./ T;

    % run algorithm (Frey's code)
    for i=1:25
        F_tmp = F;
        if add_f_noise
            % TODO randomizing zeroes seems to be bad idea
            % randomize diagonal elements
            noise_matrix = diag(2*(rand(T,1)-0.5) ./ noise_decay);
            %noise_matrix = diag(1 + poissrnd(0,T,1));
            F_tmp += noise_matrix;
            noise_decay += 10;
        end
        % do EM
        Z = F_tmp ./ (Phi * Theta); Z(F_tmp==0) = 0; % this line is correct but really slow
        if add_phi_noise
            noise_matrix = 2 * (rand(T) - 0.5) / noise_decay;
            %nonzero = Phi > 0;
            Phi += noise_matrix;% .* nonzero;
            noise_decay += 10;
        end
        Phi_tmp = Phi .* (Z * Theta');
        if add_decorr
            % ramdomized decorrelator
            tau = 0.5;
            decorrelator = Phi_tmp .* (Phi_tmp * (F == 0));
            if i < 10
                % at first, add decreasing noise
                noise = rand(T) .* (decorrelator > 0) / noise_decay;
                %noise = rand(T) / noise_decay;
                noise_decay += 5;
                decorrelator = decorrelator + noise;
            end
            Phi_tmp -= tau * decorrelator;
            Phi_tmp(Phi_tmp < 0) = 0;
        end
        Theta_tmp = Theta .* (Phi' * Z);    
        Phi = Phi_tmp ./ repmat(sum(Phi_tmp), W, 1);
        Theta = Theta_tmp ./ repmat(sum(Theta_tmp), T, 1);
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
