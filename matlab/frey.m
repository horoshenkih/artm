% problem parameters
T = 10; W = T; D = T; F = eye(T);

n_runs = 100
errors = [];
for run=1:n_runs
    % prepare RNG
    rand('seed',run)

    % prepare matrices
    Phi = rand(T); Phi = Phi ./ sum(Phi);
    Theta = rand(T); Phi = Phi ./ sum(Phi);
    %Theta = ones(T) ./ T;

    %printf('Initial matrices:\n')
    %Phi
    %Theta
    % run algorithm (Frey's code)
    for i=1:10
        Z = F ./ (Phi * Theta); Z(F==0) = 0; % this line is correct but really slow
        Phi_tmp = Phi .* (Z * Theta');
        Theta_tmp = Theta .* (Phi' * Z);    
        Phi = Phi_tmp ./ repmat(sum(Phi_tmp), W, 1);
        Theta = Theta_tmp ./ repmat(sum(Theta_tmp), T, 1);
    end

    % plot results (rounding issues in Octave)
    Phi = round(Phi*100) ./ 100;
    Theta = round(Theta*100) ./ 100;

    product = round(Phi * Theta * 100) ./ 100;

    %printf('Resulting matrices:\n')
    %Phi
    %Theta
    %printf('Error:\n')
    %product - F
    errors = [errors norm(product - F, 1)];
end
idx = find(errors > 0);
printf('Correct decomposition in %f %% cases\n', 100 * (1 - columns(idx) / n_runs));
