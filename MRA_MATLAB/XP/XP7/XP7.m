
% Sep 5, 2017, NB
% Exploring effect of weights in our cost function as a function of noise

clear all; %#ok<CLALL>
close all;
clc;

%%

L = 50;
K = 2;
sigmas = logspace(-1, 1, 11);
M = 1e6;
nrepeats = 5;


opts_MIX = struct();
opts_MIX.maxiter = 200;
opts_MIX.tolgradnorm = 1e-8;
opts_MIX.tolcost = 1e-18;
opts_MIX.verbosity = 1;

opts_MIX_new = struct();
opts_MIX_new.maxiter = 200;
opts_MIX_new.tolgradnorm = 1e-8;
opts_MIX_new.tolcost = 1e-18;
opts_MIX_new.verbosity = 1;

n_extra_inits = 1; % number of extra runs with random inits


opts_EM = struct();
opts_EM.verbosity = 1;                % Control text output
opts_EM.niter = 10000;                % Number of full-data iterations
opts_EM.niter_batch = 0; %%           % Number of batch iterations
opts_EM.batch_size = 1000;            % Size of a batch
opts_EM.tolerance = 1e-5;             % Stop when successive iterates are this close

%  !! -- setting 1 here (which is the correct prior) doesn't seem to
%        improve results with EM, and does not fix the failure mode at
%        high SNR, but it's easier to justify in comparison. (Aug. 22, 2017)
prior_EM.x0 = zeros(L, K);
prior_EM.Sigma0 = 1 * eye(L*K);


nmethods = 2; % number of methods to try

methods = { ...
   @(data, sigma, K) MRA_het_mixed_invariants_free_p(data, sigma, K, [], [], opts_MIX, [], n_extra_inits), ...
   @(data, sigma, K) MRA_het_mixed_invariants_free_p_new_coeffs(data, sigma, K, [], [], opts_MIX_new, [], n_extra_inits) %, ...
   % @(data, sigma, K) MRA_het_EM(data, sigma, K, [], prior_EM, opts_EM);
};

nmetrics = 2; % number of metrics to register for each method
% Metric 1: relative estimation error, X
% Metric 2: CPU time

metric = zeros(nmethods, nmetrics, length(sigmas), nrepeats);
    
fid = fopen('XP7_progress.txt', 'w');
origin = tic();
fprintf(fid, 'Starting: %s\r\n\r\n', datestr(now()));

for iter_sigma = 1 : length(sigmas)
    
    sigma = sigmas(iter_sigma);
        
    fprintf(fid, 'sigma = %3g, %s\r\nElapsed: %s [s]\r\n', sigma, datestr(now()), toc(origin));

    for repeat = 1 : nrepeats

        x_true = randn(L, K);
        p_true = ones(K, 1) / K; % fixed uniform, but unknown to algorithms
            
        % Make sure we have a parpool
        if isempty(gcp('nocreate'))
            parpool(30, 'IdleTimeout', 60*72); % 3 days
        end
        
        data = generate_observations_het(x_true, round(p_true*M), sigma);

        for iter_method = 1 : nmethods
        
            method = methods{iter_method};
            
            % Make sure we have a parpool
            if isempty(gcp('nocreate'))
                parpool(30, 'IdleTimeout', 60*72); % 3 days
            end
            
            % Solve from a new random initial guess.
            t = tic();
            x_est = method(data, sigma, K); % we're not getting p_est back
            t = toc(t);

            % Evaluate quality of recovery, up to permutations and shifts.
            [x_est, ~, perm] = align_to_reference_het(x_est, x_true);
%             p_est = p_est(perm);

            rel_error_X = norm(x_est - x_true) / norm(x_true);
%             tv_error_p = norm(p_est - p_true, 1) / 2;

            metric(iter_method, :, iter_sigma, repeat) = [rel_error_X, ...
                                                      ... %tv_error_p, ...
                                                      t];
                                             
        end
        
        clear data;
        
        save XP7.mat;

    end

    
end

fprintf(fid, 'Ending: %s\r\n\r\nElapsed: %s [s]\r\n', datestr(now()), toc(origin));
fclose(fid);

%%
save XP7.mat;

%%
load XP7;

markers = {'.', 'o', 'x'};

clf;
ColOrd = get(gca, 'ColorOrder');

jitter = 10.^(randn(1, nrepeats)/60);

subplot(2, 1, 1);
hold all;
for iter_method = nmethods : -1 : 1
    loglog(sigmas'*jitter, squeeze(metric(iter_method, 1, :, :)), markers{iter_method}, 'Color', ColOrd(iter_method, :));
end
title('Relative estimation error of the signals');
xlabel('Noise level \sigma');
% legend('Mixed invariants', 'EM');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
grid on;

% ylim([1e-4, 10^(1)]); %% !! check if change XP parameters

plotted_sigmas = sigmas'*jitter;
xlim([min(plotted_sigmas(:)), max(plotted_sigmas(:))]);

subplot(2, 1, 2);
hold all;
for iter_method = nmethods : -1 : 1
    loglog(sigmas'*jitter, squeeze(metric(iter_method, 2, :, :)), markers{iter_method}, 'Color', ColOrd(iter_method, :));
end
title('Computation time (seconds)');
xlabel('Noise level \sigma');
% legend('Mixed invariants', 'EM');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
grid on;

plotted_sigmas = sigmas'*jitter;
xlim([min(plotted_sigmas(:)), max(plotted_sigmas(:))]);

%%
savefig('XP7.fig');
pdf_print_code(gcf, 'XP7.pdf');

