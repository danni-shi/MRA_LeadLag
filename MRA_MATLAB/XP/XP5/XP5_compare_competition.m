
% July 24, 2017, NB
% Comparing moment demixing to competitors with heterogeneity

clear all; %#ok<CLALL>
close all;
clc;

%%

L = 50;
K = 3;
sigma = 1;
Ms = unique(round(logspace(1, 6, 11)));
nrepeats = 10;


opts_MIX = struct();
opts_MIX.maxiter = 200;
opts_MIX.tolgradnorm = 1e-8;
opts_MIX.tolcost = 1e-18;
opts_MIX.verbosity = 1;

opts_EM = struct();
opts_EM.verbosity = 0;                % Control text output
opts_EM.niter = 10000;                % Number of full-data iterations
opts_EM.niter_batch = 0; %%           % Number of batch iterations
opts_EM.batch_size = 1000;            % Size of a batch
opts_EM.tolerance = 1e-5;             % Stop when successive iterates are this close

nmethods = 2; % number of methods to try
methods = {@(data, sigma, K) MRA_het_mixed_invariants_free_p(data, sigma, K, [], [], opts_MIX), ...
           @(data, sigma, K) MRA_het_EM(data, sigma, K, [], [], opts_EM)};

nmetrics = 2; % number of metrics to register for each method
% Metric 1: relative estimation error, X
% Metric 2: CPU time

metric = zeros(nmethods, nmetrics, length(Ms), nrepeats);
    
fid = fopen('XP5_progress.txt', 'w');
origin = tic();
fprintf(fid, 'Starting: %s\r\n\r\n', datestr(now()));

for iter_M = 1 : length(Ms)
    
    M = Ms(iter_M);
        
    fprintf(fid, 'M = %3d, %s\r\nElapsed: %s [s]\r\n', M, datestr(now()), toc(origin));

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

            metric(iter_method, :, iter_M, repeat) = [rel_error_X, ...
                                                      ... %tv_error_p, ...
                                                      t];
                                             
        end
        
        clear data;

    end

    save XP5.mat;
    
end

fprintf(fid, 'Ending: %s\r\n\r\nElapsed: %s [s]\r\n', datestr(now()), toc(origin));
fclose(fid);

%%
save XP5.mat;

%%
load XP5;

clf;
ColOrd = get(gca, 'ColorOrder');

subplot(2, 1, 1);
hold all;
for iter_method = 1 : nmethods
    loglog(Ms, squeeze(metric(iter_method, 1, :, :)), '.-', 'Color', ColOrd(iter_method, :));
end
title('Average relative estimation error of the signals');
xlabel('# of measurements M');
legend('Mixed invariants', 'EM');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

subplot(2, 1, 2);
hold all;
for iter_method = 1 : nmethods
    loglog(Ms, squeeze(metric(iter_method, 2, :, :)), '.-', 'Color', ColOrd(iter_method, :));
end
title('Average computation time');
xlabel('# of measurements M');
% legend('Mixed invariants', 'EM');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

%%
savefig('XP5.fig');
pdf_print_code(gcf, 'XP5.pdf');

