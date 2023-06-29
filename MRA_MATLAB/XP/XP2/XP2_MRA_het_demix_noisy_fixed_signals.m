
% July 10, 2017, NB
% Fixed L (signal length) and K (heterogeneity) -- variable sigma and M

clear all; %#ok<CLALL>
close all;
clc;

%% Fix the data size

L = 50;
K = 3;

%% Parameters to sweep
sigmas = logspace(-2, 1, 25); % 25
Ms = unique(round(logspace(0, 4, 17))); % 17
nrepeats = 30; % 30
nmetrics = 3;
metric = zeros(nmetrics, length(sigmas), length(Ms), nrepeats);
% Metric 1: relative estimation error
% Metric 2: objective value reached
% Metric 3: CPU time

%% Loop

opts = struct();
opts.maxiter = 200;
opts.tolgradnorm = 1e-10;
opts.tolcost = 1e-18;
    
fid = fopen('XP2_progress.txt', 'a');
origin = tic();
fprintf(fid, 'Starting: %s\r\n\r\n', datestr(now()));

for iter_sigma = 1 : length(sigmas)
    
    sigma = sigmas(iter_sigma);
        
    fprintf(fid, 'sigma = %3g, %s\r\nElapsed: %s [s]\r\n', sigma, datestr(now()), toc(origin));
    
    for iter_M = 1 : length(Ms)
        
        M = Ms(iter_M);
        
%         if mod(iter_M, 10) == 0
            fprintf(fid, '\tM = %6d, %s\r\n', M, datestr(now()));
%         end
        
        parfor repeat = 1 : nrepeats
        
            % Generate new ground truth and data
            x_true = randn(L, K);
            
            data = generate_observations_het(x_true, M*ones(K, 1), sigma);
        
            % Solve from a random initial guess.
            t = tic();
            [x_est, problem] = MRA_het_mixed_invariants(data, sigma, K, [], opts);
            t = toc(t);

            % Evaluate quality of recovery, up to permutations and shifts.
            x_est = align_to_reference_het(x_est, x_true);
            relative_error = norm(x_est - x_true) / norm(x_true);
            metric(:, iter_sigma, iter_M, repeat) = [relative_error, getCost(problem, x_est), t];
            
        end
        
    end
    
    save XP2.mat;
    
end

fprintf(fid, 'Ending: %s\r\n\r\nElapsed: %s [s]\r\n', datestr(now()), toc(origin));
fclose(fid);

%%
save XP2.mat;

%%
figure(1);
clf;
metric1 = squeeze(metric(1, :, :, :));
metric2 = squeeze(metric(2, :, :, :));
metric3 = squeeze(metric(3, :, :, :));


subplot(3, 1, 1);
imagesc(log10(sigmas), log10(Ms), mean(log10(metric1), 3)');
xlabel('log_{10}(\sigma)');
ylabel('log_{10}(M)');
title('Geometric mean estimation error (log_{10})');
set(gca, 'YDir', 'normal');
colorbar;


subplot(3, 1, 2);
imagesc(log10(sigmas), log10(Ms), mean(log10(metric2), 3)');
xlabel('log_{10}(\sigma)');
ylabel('log_{10}(M)');
title('Geometric mean objective value reached (log_{10})');
set(gca, 'YDir', 'normal');
colorbar;


subplot(3, 1, 3);
imagesc(log10(sigmas), log10(Ms), mean(log10(metric3), 3)');
xlabel('log_{10}(\sigma)');
ylabel('log_{10}(M)');
title('Geometric mean computation time (log_{10})');
set(gca, 'YDir', 'normal');
colorbar;

set(gcf, 'Color', 'w');

%%
savefig('XP2.fig');
pdf_print_code(gcf, 'XP2.pdf');