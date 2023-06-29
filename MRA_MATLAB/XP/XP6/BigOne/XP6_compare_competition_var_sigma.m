
% July 25, 2017, NB
% Comparing moment demixing to competitors with heterogeneity, variable
% sigma

clear all; %#ok<CLALL>
close all;
clc;

%%

L = 50;
K = 2;
sigmas = logspace(-1, 1, 11);
M = 1e6;
nrepeats = 20;


opts_MIX = struct();
opts_MIX.maxiter = 200;
opts_MIX.tolgradnorm = 1e-8;
opts_MIX.tolcost = 1e-18;
opts_MIX.verbosity = 1;

opts_EM = struct();

nmethods = 2; % number of methods to try
methods = {@(data, sigma, K) MRA_het_mixed_invariants_free_p(data, sigma, K, [], [], opts_MIX), ...
           @(data, sigma, K) MRA_het_EM(data, sigma, K, [], [], opts_EM)};

nmetrics = 2; % number of metrics to register for each method
% Metric 1: relative estimation error, X
% Metric 2: CPU time

metric = zeros(nmethods, nmetrics, length(sigmas), nrepeats);
    
fid = fopen('XP6_progress.txt', 'w');
origin = tic();
fprintf(fid, 'Starting: %s\r\n\r\n', datestr(now()));

for iter_sigma = 1 : length(sigmas)
    
    sigma = sigmas(iter_sigma);
        
    fprintf(fid, 'sigma = %3g, %s\r\nElapsed: %s [s]\r\n', sigma, datestr(now()), toc(origin));

    for repeat = 1 : nrepeats

        x_true = randn(L, K);
        p_true = ones(K, 1) / K; % fixed uniform, but unknown to algorithms
        
        data = generate_observations_het(x_true, round(p_true*M), sigma);

        for iter_method = 1 : nmethods
        
            method = methods{iter_method};
            
            % Solve from a new random initial guess.
            t = tic();
            x_est = method(data, sigma, K); % we're not getting p_est back
            t = toc(t);

            % Evaluate quality of recovery, up to permutations and shifts.
            [x_est, ~, perm] = align_to_reference_het(x_est, x_true);
%             p_est = p_est(perm);

            rel_error_X = norm(x_est - x_true) / norm(x_true);
%             tv_error_p = norm(p_est - p_true, 1) / 2;

sum((x_true - x_est).^2)

            metric(iter_method, :, iter_sigma, repeat) = [rel_error_X, ...
                                                      ... %tv_error_p, ...
                                                      t];
                                             
        end
        
        clear data;

    end

    save XP6.mat;
    
end

fprintf(fid, 'Ending: %s\r\n\r\nElapsed: %s [s]\r\n', datestr(now()), toc(origin));
fclose(fid);

%%
save XP6.mat;

%%
load XP6;

clf;
ColOrd = get(gca, 'ColorOrder');

subplot(2, 1, 1);
hold all;
for iter_method = 1 : nmethods
    loglog(sigmas, squeeze(metric(iter_method, 1, :, :)), '.-', 'Color', ColOrd(iter_method, :));
end
title('Relative estimation error of the signals');
xlabel('Noise level \sigma');
% legend('Mixed invariants', 'EM');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
grid on;

subplot(2, 1, 2);
hold all;
for iter_method = 1 : nmethods
    loglog(sigmas, squeeze(metric(iter_method, 2, :, :)), '.-', 'Color', ColOrd(iter_method, :));
end
title('Computation time');
xlabel('Noise level \sigma');
% legend('Mixed invariants', 'EM');
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
grid on;

%%
savefig('XP6.fig');
pdf_print_code(gcf, 'XP6.pdf');

