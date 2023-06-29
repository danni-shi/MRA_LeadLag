function [X_est, problem] = MRA_het_mixed_invariants(data, sigma, K, X0, opts, w, nextrainits)

    [N, M] = size(data);
    
    if ~exist('w', 'var') || isempty(w)
        w = ones(M, 1);
    end
    assert(length(w) == M, 'w must have length M');
    assert(all(w >= 0), 'w must be nonnegative');
    w = w / sum(w);
    
    if ~exist('nextrainits', 'var') || isempty(nextrainits)
        nextrainits = 0;
    end

    [mumix, psmix, Bmix] = invariants_from_data_no_debias(data, w);
    
    manifold = euclideanfactory(N, K);
    problem.M = manifold;
    
    problem.costgrad = @(X, store) costgrad(X, Bmix, psmix, mumix, sigma, manifold, store);
    
    if ~exist('X0', 'var') || isempty(X0)
        X0 = randn(N, K);
    end
    
    if ~exist('opts', 'var') || isempty(opts)
        opts.maxiter = 200;
        opts.tolgradnorm = 1e-8;
        opts.tolcost = 1e-10;
    end
    
    warning('off', 'manopt:getHessian:approx');
    
    % Run once with the given initialization (could be random)
    [X_est, cost] = trustregions(problem, X0, opts);
    
    % Also run an extra number of random inits and keep the best result.
    for init = 1 : nextrainits
        [X_est_bis, cost_bis] = trustregions(problem, [], opts);
        if cost_bis < cost
            X_est = X_est_bis;
            cost = cost_bis;
        end
    end
    
    warning('on', 'manopt:getHessian:approx');

end


function [f, gradf, store] = costgrad(X, Bmix, psmix, mumix, sigma, manifold, store)

    if ~isfield(store, 'cost') || ~isfield(store, 'grad')

        [N, K] = size(X);
        
        % Assume uniform mixing probability (could be fixed to something
        % else too.)
        p = ones(K, 1) / K;

        % Bias matrix pattern for Bmix
        A = eye(N);
        A(1, :) = A(1, :) + 1;
        if isreal(X)
            A(:, 1) = A(:, 1) + 1;
        end

        FX = fft(X);
        absFXsq = abs(FX).^2;

        B = zeros(N, N, K);

        M1 = mean(X*p);
        M2 = absFXsq*p + N*sigma^2*ones(N, 1);
        M3 = N^2*sigma^2*mumix*A; % Choose whether to use M1 or mumix here.
        for k = 1 : K
            y = FX(:, k);
            B(:, :, k) = (y*y') .* circulant(y) ; % = bispectrum_from_signal(X(:, k));
            M3 = M3 + p(k) * B(:, :, k);
        end

        % Precompute, because costs O(N^2) and needed K times, for a total
        % of O(KN^2), which is the overall complexity of this function.
        M3_min_Bmix = M3 - Bmix;

        a1 = N^2; % would be 1 but we work with the mean instead of DC component.
        a2 = (N*(2+sigma^2))^-1;
        a3 = (N^2*(3+sigma^4))^-1;
        
        
        scale = 3+sigma^4; %N*sigma^6; % to normalize coefficient a3 to scale*a3 = 1/N^2, which is our previous weight.
        
        
        
        f = .5*(a1*(M1 - mumix)^2  +  ...
                a2*norm(M2 - psmix, 2)^2  +  ...
                a3*norm(M3_min_Bmix, 'fro')^2 );

        f = scale * f;
            
        store.cost = f;
        
        

        gradX = (a1/N)*(M1-mumix)*repmat(p', [N, 1]) + ...
                2*N*a2*ifft( ((M2-psmix)*p') .* FX );

        for k = 1 : K

            y = FX(:, k);

            gradX(:, k) = gradX(:, k) + a3 * p(k) * DBx_adj(y, M3_min_Bmix);

        end

        if isreal(X)
            gradX = real(gradX);
        end

        gradf = scale * gradX;

        gradf = manifold.egrad2rgrad(X, gradf); % identity if no constraints

        store.grad = gradf;
        
        
    else
        
        f = store.cost;
            
        gradf = store.grad;
        
    end

end

% Adjoint of the differential of bispectrum_from_signal applied to a matrix
% W, where the input y is the fft of x.
function z = DBx_adj(y, W)
    % y = fft(x)
    % B(x) = yy' .* circulant(y)
    % F = fft(eye(N)); F' == N*ifft(eye(N))
    N = length(y);
    H = W .* circulant(conj(y));
    z = N*ifft( circulantadj(W.*conj(y*y')) + (H+H')*y );
end

