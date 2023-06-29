function [data, shifts, classes] = generate_observations_het(x_true, Ms, sigma, shift)
% Generate MRA observations with heterogeneity.
% x_true is a matrix of size N x K, containing the K true signals of length N
% Ms is a vector of integers of length K
% sigma is the noise standard deviation
% data is a matrix of size N x sum(Ms) such that Ms(k) of the columns of data
% are circularly-shifted, noisy measurements of x_true(:, k).

	[N, K] = size(x_true);
	
    assert(length(Ms) == K, 'Ms should be a vector of length K.');
	
	cumsumMs = [0 ; cumsum(Ms(:))];
    sumMs = cumsumMs(end);
	
	data = zeros(N, sumMs);
	shifts = zeros(sumMs, 1);
    classes = zeros(sumMs, 1);
    for k = 1 : K
        range = (cumsumMs(k) + 1) : cumsumMs(k+1);
        [data(:, range), shifts(range)] = generate_observations(x_true(:, k), Ms(k), sigma, shift);
        classes(range) = k;
    end

    % Randomly permute the observations
    perm = randperm(sumMs);
    data = data(:, perm);
    shifts = shifts(perm);
    classes = classes(perm);
    
end
