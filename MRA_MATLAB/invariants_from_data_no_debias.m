function [mean_est, P_est, B_est] = invariants_from_data_no_debias(X, w)
% See invariants_from_data -- but no debiasing is attempted.
% The bispectrum is computed on the data directly (no centering).
% w is a weight vector: nonnegative entries, length = size(X, 2)

    [N, M] = size(X);
    
    if ~exist('w', 'var') || isempty(w)
        w = ones(M, 1);
    end
    w = w(:);
    assert(length(w) == M, 'w must have length M = size(X, 2)');
    assert(all(w >= 0), 'w must be nonnegative');
    w = w / sum(w);
    
    %% Estimate the mean and subtract it from the observations.
    %  This also gives an estimate of the first component of the fft of x.
    %  Centering the observations helps for the bispectrum estimation part.
    mean_est = mean(X, 1) * w;
    
    %% Prepare fft's
    X_fft = fft(X);

    %% Estimate the power spectrum (gives estimate of modulus of fft of x).
    P_est = abs(X_fft).^2 * w;

    %% Estimate the bispectrum
    if nargout >= 3
        
        B_est = zeros(N);
        parfor m = 1 : M
            xm_fft = X_fft(:, m);
            Bm = (xm_fft*xm_fft') .* circulant(xm_fft);
            B_est = B_est + w(m) * Bm;
        end
        
    end

end
