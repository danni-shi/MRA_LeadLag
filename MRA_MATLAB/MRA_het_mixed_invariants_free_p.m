function [X, p, problem] = MRA_het_mixed_invariants_free_p(data, sigma, K, X0, p0, opts, w, nextrainits)

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
    
    tuple.X = euclideanfactory(N, K);
    tuple.p = multinomialfactory(K);
    manifold = productmanifold(tuple);
    problem.M = manifold;
    
    problem.costgrad = @(Xp, store) costgrad(Xp, Bmix, psmix, mumix, sigma, manifold, store);
    
    if ~exist('X0', 'var') || isempty(X0)
        X0 = randn(N, K);
    end
    if ~exist('p0', 'var') || isempty(p0)
        p0 = ones(K, 1) / K;
    end
    XP0.X = X0;
    XP0.p = p0;
    
    if ~exist('opts', 'var') || isempty(opts)
        opts.maxiter = 200;
        opts.tolgradnorm = 1e-8;
        opts.tolcost = 1e-10;
    end
    
    warning('off', 'manopt:getHessian:approx');
    
    % Run once with the given initialization (could be random)
    [XP_est, cost, info] = trustregions(problem, XP0, opts);
    
    
    % Also run an extra number of random inits and keep the best result.
    for init = 1 : nextrainits
        % only rerun if previous round ends at maximum iteration
        iter_list = [info.iter];
        if length(iter_list) > opts.maxiter
            [XP_est_bis, cost_bis, info] = trustregions(problem, [], opts);
            if cost_bis < cost
                XP_est = XP_est_bis;
                cost = cost_bis;
            end
        end
    end
    warning('on', 'manopt:getHessian:approx');
    
    X = XP_est.X;
    p = XP_est.p;

end


function [f, gradf, store] = costgrad(Xp, Bmix, psmix, mumix, sigma, manifold, store)

    if ~isfield(store, 'cost') || ~isfield(store, 'grad')

        X = Xp.X;
        p = Xp.p;

        [N, K] = size(X);

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

        gradp = (a1/N)*(M1-mumix)*sum(X)' + ...
                a2*absFXsq'*(M2 - psmix);

        for k = 1 : K

            y = FX(:, k);

            gradX(:, k) = gradX(:, k) + a3 * p(k) * DBx_adj(y, M3_min_Bmix);

            gradp(k) = gradp(k) + a3 * inner(M3_min_Bmix, B(:, :, k));

        end

        if isreal(X)
            gradX = real(gradX);
        end

        gradf.X = scale * gradX;
        gradf.p = scale * gradp;

        gradf = manifold.egrad2rgrad(Xp, gradf);

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

% Real inner product for complex matrices
function v = inner(A, B)
    v = real(A(:)'*B(:));
end



% function [X, p, problem] = MRA_het_mixed_invariants_free_p(data, sigma, K, X0, p0, opts, w, nextrainits)
% 
%     [N, M] = size(data);
%     
%     if ~exist('w', 'var') || isempty(w)
%         w = ones(M, 1);
%     end
%     assert(length(w) == M, 'w must have length M');
%     assert(all(w >= 0), 'w must be nonnegative');
%     w = w / sum(w);
%     
%     if ~exist('nextrainits', 'var') || isempty(nextrainits)
%         nextrainits = 0;
%     end
% 
%     [mumix, psmix, Bmix] = invariants_from_data_no_debias(data, w);
% 
%     tuple.X = euclideanfactory(N, K);
%     tuple.p = multinomialfactory(K);
%     manifold = productmanifold(tuple);
%     problem.M = manifold;
%     
%     problem.costgrad = @(Xp, store) costgrad(Xp, Bmix, psmix, mumix, sigma, manifold, store);
%     
%     if ~exist('X0', 'var') || isempty(X0)
%         X0 = randn(N, K);
%     end
%     if ~exist('p0', 'var') || isempty(p0)
%         p0 = ones(K, 1) / K;
%     end
%     XP0.X = X0;
%     XP0.p = p0;
%     
%     if ~exist('opts', 'var') || isempty(opts)
%         opts.maxiter = 200;
%         opts.tolgradnorm = 1e-8;
%         opts.tolcost = 1e-10;
%     end
%     
%     warning('off', 'manopt:getHessian:approx');
%     
%     % Run once with the given initialization (could be random)
%     [XP_est, cost] = trustregions(problem, XP0, opts);
%     
%     % Also run an extra number of random inits and keep the best result.
%     for init = 1 : nextrainits
%         [XP_est_bis, cost_bis] = trustregions(problem, [], opts);
%         if cost_bis < cost
%             XP_est = XP_est_bis;
%             cost = cost_bis;
%         end
%     end
%     
%     warning('on', 'manopt:getHessian:approx');
%     
%     X = XP_est.X;
%     p = XP_est.p;
% 
% end
% 
% 
% function [f, gradf, store] = costgrad(Xp, Bmix, psmix, mumix, sigma, manifold, store)
% 
%     if ~isfield(store, 'cost') || ~isfield(store, 'grad')
% 
%         X = Xp.X;
%         p = Xp.p;
% 
%         [N, K] = size(X);
% 
%         % Bias matrix pattern for Bmix
%         A = eye(N);
%         A(1, :) = A(1, :) + 1;
%         if isreal(X)
%             A(:, 1) = A(:, 1) + 1;
%         end
% 
%         FX = fft(X);
%         absFXsq = abs(FX).^2;
% 
%         B = zeros(N, N, K);
% 
%         M1 = mean(X*p);
%         M2 = absFXsq*p + N*sigma^2*ones(N, 1);
%         M3 = N^2*sigma^2*mumix*A; % Choose whether to use M1 or mumix here.
%         for k = 1 : K
%             y = FX(:, k);
%             disp(k);
%             B(:, :, k) = (y*y') .* circulant(y) ; % = bispectrum_from_signal(X(:, k));
%             M3 = M3 + p(k) * B(:, :, k);
%         end
% 
%         % Precompute, because costs O(N^2) and needed K times, for a total
%         % of O(KN^2), which is the overall complexity of this function.
%         M3_min_Bmix = M3 - Bmix;
% 
%         a1 = N^2; % would be 1 but we work with the mean instead of DC component.
%         a2 = (N*(2+sigma^2))^-1;
%         a3 = (N^2*(3+sigma^4))^-1;
%         
%         
%         scale = 3+sigma^4; %N*sigma^6; % to normalize coefficient a3 to scale*a3 = 1/N^2, which is our previous weight.
%         
%         
%         
%         f = .5*(a1*(M1 - mumix)^2  +  ...
%                 a2*norm(M2 - psmix, 2)^2  +  ...
%                 a3*norm(M3_min_Bmix, 'fro')^2 );
% 
%         f = scale * f;
%             
%         store.cost = f;
%         
%         
% 
%         gradX = (a1/N)*(M1-mumix)*repmat(p', [N, 1]) + ...
%                 2*N*a2*ifft( ((M2-psmix)*p') .* FX );
% 
%         gradp = (a1/N)*(M1-mumix)*sum(X)' + ...
%                 a2*absFXsq'*(M2 - psmix);
% 
%         for k = 1 : K
% 
%             y = FX(:, k);
% 
%             gradX(:, k) = gradX(:, k) + a3 * p(k) * DBx_adj(y, M3_min_Bmix);
% 
%             gradp(k) = gradp(k) + a3 * inner(M3_min_Bmix, B(:, :, k));
% 
%         end
% 
%         if isreal(X)
%             gradX = real(gradX);
%         end
% 
%         gradf.X = scale * gradX;
%         gradf.p = scale * gradp;
% 
%         gradf = manifold.egrad2rgrad(Xp, gradf);
% 
%         store.grad = gradf;
%         
%         
%     else
%         
%         f = store.cost;
%             
%         gradf = store.grad;
%         
%     end
% 
% end
% 
% % Adjoint of the differential of bispectrum_from_signal applied to a matrix
% % W, where the input y is the fft of x.
% function z = DBx_adj(y, W)
%     % y = fft(x)
%     % B(x) = yy' .* circulant(y)
%     % F = fft(eye(N)); F' == N*ifft(eye(N))
%     N = length(y);
%     H = W .* circulant(conj(y));
%     z = N*ifft( circulantadj(W.*conj(y*y')) + (H+H')*y );
% end
% 
% % Real inner product for complex matrices
% function v = inner(A, B)
%     v = real(A(:)'*B(:));
% end
