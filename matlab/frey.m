T = 3; W = T; D = T; F = eye(T);
Phi =   [0.2755, 0.9333, 0.6804; 0.9871, 0.3484, 0.3554; 0.7950, 0.2091, 0.4384]
Theta = [0.3380, 0.6988, 0.6354; 0.7343, 0.0899, 0.5253; 0.9855, 0.4707, 0.1413]
for i=1:6
    Z = F ./ (Phi * Theta); Z(F==0) = 0; % this line is correct but really slow
    Phi_tmp = Phi .* (Z * Theta');
    Theta_tmp = Theta .* (Phi' * Z);    
    Phi = Phi_tmp ./ repmat(sum(Phi_tmp), W, 1);
    Theta = Theta_tmp ./ repmat(sum(Theta_tmp), T, 1);
end

% rounding issues in Octave
round(Phi*100) ./ 100
round(Theta*100) ./ 100

round(Phi * Theta * 100) ./ 100
