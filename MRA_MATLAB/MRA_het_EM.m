function [x, mylog] = MRA_het_EM(data, sigma, K, x_init, prior, options)
% K - number of clusters
% prior is a structure with two fields:
%  prior.x0 is a matrix of size NxK
%  prior.Sigma0 is a matrix of size NK x NK (covariance for the vectorized
%  form of x0)

    % data contains M observations as columns, each of length N
    [N, M] = size(data);
    
    % Set local defaults here
    localdefaults.verbosity = 0;                % Control text output
    localdefaults.niter = 10000;                % Number of full-data iterations
    if M > 3000
        localdefaults.niter_batch = 1000;       % Number of batch iterations
    else
        localdefaults.niter_batch = 0;
    end
    localdefaults.batch_size = 1000;            % Size of a batch
    localdefaults.tolerance = 1e-5;             % Relative tolerance for stopping criterion

    % Merge local defaults with user options, if any.
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % Default prior if omitted (almost information-less)
    if ~exist('prior', 'var') || isempty(prior)
        prior.x0 = zeros(N, K);
        prior.Sigma0 = 1e9 * eye(N*K);
    end
    
    
    % Initial guess of the signal (TODO: use prior for this?)
    if ~exist('x_init', 'var') || isempty(x_init)
        if isreal(data)
            x_init = randn(N, K);
        else
            x_init = randn(N, K) + 1i*randn(N, K);
        end
    end
    assert(all(size(x_init) == [N, K]), ...
           'Initial guess x_init must have size NxK.');
    
    % Initialize
    x = x_init;
    
    % In practice, we iterate with the DFT of the signal x
    fftx = fft(x);
    
    % Precomputations on the observations
    fftdata = fft(data);
    sqnormdata = repmat(sum(abs(data).^2, 1), N, 1);
    
    % Get started with batch iterations: these only consider a subsample of
    % the observations.
    if options.niter_batch > 0
        
        for iter = 1 : options.niter_batch
            
            % Take a random sample of the dataset (with replacement)
            sample = randi(M, options.batch_size, 1);
            
            fftx_new = EM_iteration(fftx, fftdata(:, sample), ...
                                   sqnormdata(:, sample), sigma, K, prior);
            
            fftx = fftx_new;
            
        end
        
        if options.verbosity >= 1
            fprintf('EM batch iterations: %5d\n', iter);
        end
        
    end
    
    
    % In any case, finish with full passes on the data
    for iter = 1 : options.niter
        
        [fftx_new, W] = EM_iteration(fftx, fftdata, sqnormdata, sigma, K, prior);

        x_current = ifft(fftx);
        x_new = ifft(fftx_new);
        if norm(align_to_reference_het(x_current, x_new) - x_new, 'fro') ...
                                                      < K*options.tolerance
            break;
        end

        fftx = fftx_new;

    end
    
    if options.verbosity >= 1
        fprintf('EM full  iterations: %5d\n', iter);
    end
    
    x = ifft(fftx_new);
    
    mylog = struct();
    mylog.W = W;

end


% Execute one iteration of EM with current estimate of the DFT of the
% signal given by fftx, and DFT's of the observations stored in fftX, and
% squared 2-norms of the observations stored in sqnormX, and noise level
% sigma.
function [fftx_new, W] = EM_iteration(fftx, fftdata, sqnormdata, sigma, K, prior)

    [N, M] = size(fftdata);
    
    C = zeros(N, M, K);
    T = zeros(N, M, K);
    for k = 1 : K
        C(:, :, k) = ifft(bsxfun(@times, conj(fftx(:, k)), fftdata));
        T(:, :, k) = -(sqnormdata + norm(ifft(fftx(:, k)), 2)^2 - 2*C(:, :, k))/(2*sigma^2);
    end
    
    % Compute maximum entry of T in first and last dimension.
    Tmax = max(max(T, [], 3), [], 1);
    
    % Subtract max extries computed and exponentiate: this gives
    % probabilities up to a constant multiplier along first and last
    % dimension.
    W = exp(bsxfun(@minus, T, Tmax));
    
    % Normalize the probabilities so they sum to 1 over first and last
    % dimension.
    W = bsxfun(@times, W, 1./sum(sum(W, 3), 1));
        
    b = zeros(N, K);
    diags = zeros(N, K);
    for k = 1 : K
        diags(:, k) = sum(sum(W(:, :, k))) / sigma^2; % squeeze
        b(:, k) = ifft(sum(conj(fft(W(:, :, k))).*fftdata, 2)) / sigma^2;
        
        % !!
        if sum(sum(W(:, :, k))) < 1e-6
            warning('Careful now');
%             keyboard;
%                 bad_ks(end+1) = k;
%             x_new(:, k) = randn(N, 1);
%             break;
        end
        
    end
    
%     fprintf('EM: sum of weights for each class:\n\t');
%     fprintf('%g\t', diags(1, :));
%     fprintf('\n');
    
    S = inv(prior.Sigma0) + diag(diags(:));
    b = b(:) + prior.Sigma0 \ prior.x0(:);
    x_new = reshape(S \ b, [N, K]);
    
%     x_new(:, bad_ks) = randn(N, numel(bad_ks));
    
    fftx_new = fft(x_new);
    
end
