% problem parameters
T = 10; W = T; D = T; F = eye(T);

% prepare RNG
rand('seed',42)

% prepare matrices
Phi = rand(T); Phi = Phi ./ sum(Phi);
Theta = ones(T) ./ T;

printf("Initial matrices:\n")
Phi
Theta
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

printf("Resulting matrices:\n")
Phi
Theta
product
