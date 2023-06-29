
% July 13, 2017, NB
% Current code is noiseless regime rather than infinite data regime, in
% that there is no bias term added to the (perfect) mixed moments.
%
% Need to think about how to generate random p's.

clear all; %#ok<CLALL>
close all;
clc;

%%

Ls = 1:1:50;
Ks = 1:7;
nrepeats = 30;
nmetrics = 5;
metric = zeros(nmetrics, length(Ls), length(Ks), nrepeats);
% Metric 1: relative estimation error, X
% Metric 2: objective value reached
% Metric 3: CPU time
% Metric 4: total variation error, p
% Metric 5: min(p_true)

opts = struct();
opts.maxiter = 200;
opts.tolgradnorm = 1e-10;
opts.tolcost = 1e-18;
    
fid = fopen('XP4_progress.txt', 'w');
origin = tic();
fprintf(fid, 'Starting: %s\r\n\r\n', datestr(now()));

for iter_K = 1 : length(Ks)
    
    K = Ks(iter_K);
        
    fprintf(fid, 'K = %3d, %s\r\nElapsed: %s [s]\r\n', K, datestr(now()), toc(origin));
    
    for iter_L = 1 : length(Ls)
        
        L = Ls(iter_L);
        
%         if mod(iter_L, 10) == 0
            fprintf(fid, '\tL = %3d, %s\r\n', L, datestr(now()));
%         end
        
        x_true = randn(L, K);
        
        p_true = rand(K, 1);
        p_true = p_true / sum(p_true);
        
        parfor repeat = 1 : nrepeats
        
            % Solve from a new random initial guess.
            t = tic();
            [x_est, p_est, problem] = MRA_het_mixed_invariants_free_p(x_true, 0, K, [], [], opts, p_true);
            t = toc(t);

            % Evaluate quality of recovery, up to permutations and shifts.
            [x_est, ~, perm] = align_to_reference_het(x_est, x_true);
            p_est = p_est(perm);

            rel_error_X = norm(x_est - x_true) / norm(x_true);
            tv_error_p = norm(p_est - p_true, 1) / 2;

            metric(:, iter_L, iter_K, repeat) = [rel_error_X, ...
                                                 getCost(problem, struct('X', x_est, 'p', p_est)), ...
                                                 t, ...
                                                 tv_error_p, ...
                                                 min(p_true)];
            
        end
    
        save XP4.mat;
        
    end
    
end

fprintf(fid, 'Ending: %s\r\n\r\nElapsed: %s [s]\r\n', datestr(now()), toc(origin));
fclose(fid);

%%
save XP4.mat;

%%
clear all;
close all;
clc;

load XP4;

% Investigation of bad point K : 4, L : 49:
% a = squeeze(metric([1, 2], 49, 4, :))'; loglog(a(:, 2), a(:, 1), '.')

figure(1);
clf;
metric1 = squeeze(metric(1, :, :, :));
metric2 = squeeze(metric(2, :, :, :));
metric3 = squeeze(metric(3, :, :, :));
metric4 = squeeze(metric(4, :, :, :));
metric5 = squeeze(metric(5, :, :, :));
% subplot(3, 1, 1);
% imagesc(Ls, Ks, log10(median(metric1, 3))');
% xlabel('L');
% ylabel('K');
% title('Median log10 of recovery error');
% set(gca, 'YDir', 'normal');
% colorbar;
% axis equal;
% axis tight;
subplot(4, 1, 1);
% imagesc(Ls, Ks, log10(min(metric2, [], 3))');
imagesc(Ls, Ks, mean(metric2 <= 1e-16, 3)');
% xlabel('L');
ylabel('K');
% title('Smallest objective value attained in log10');
% title('Fraction of repetitions that lead to cost value \leq 10^{-16}');
title('Fraction of initializations leading to optimality');
set(gca, 'YDir', 'normal');
colorbar;
axis equal;
axis tight;

subplot(4, 1, 2);
Q = zeros(length(Ls), length(Ks));
for iter_L = 1 : length(Ls)
    for iter_K = 1 : length(Ks)
        q = find(metric2(iter_L, iter_K, :) <= 1e-16);
        z = max(squeeze(metric1(iter_L, iter_K, q)));
        if isempty(z)
            z = 1; % if no optimum found, return 0: relative error is 1
        end
        Q(iter_L, iter_K) = z;
    end
end
imagesc(Ls, Ks, log10(Q'));
% xlabel('L');
ylabel('K');
% title('Smallest objective value attained in log10');
% title('Largest relative estimation error on X, among repetitions with cost \leq 10^{-16}');
title('Largest relative estimation error among computed optima');
set(gca, 'YDir', 'normal');
colorbar;
axis equal;
axis tight;

subplot(4, 1, 3);
Q = zeros(length(Ls), length(Ks));
for iter_L = 1 : length(Ls)
    for iter_K = 1 : length(Ks)
        q = find(metric2(iter_L, iter_K, :) <= 1e-16);
        z = max(squeeze(metric4(iter_L, iter_K, q))); % only change is here
        if isempty(z)
            z = 1; % max error is 1
        end
        Q(iter_L, iter_K) = z;
    end
end
imagesc(Ls, Ks, log10(Q'));
% xlabel('L');
ylabel('K');
% title('Smallest objective value attained in log10');
% title('Largest total variation error on p, among repetitions with cost \leq 10^{-16}');
title('Largest total variation error on w among computed optima');
set(gca, 'YDir', 'normal');
colorbar;
axis equal;
axis tight;

subplot(4, 1, 4);
imagesc(Ls, Ks, mean(log10(metric3), 3)');
xlabel('L');
ylabel('K');
% title('Smallest objective value attained in log10');
% title('Average computation time (arithmetic mean, log10 scale)');
title('Average computation time (arithmetic mean, log_{10} scale)');
set(gca, 'YDir', 'normal');
colorbar;
axis equal;
axis tight;

set(gcf, 'Color', 'w');

t = linspace(min(Ls), max(Ls), 101);
for sp = 1 : 4
    subplot(4, 1, sp); hold all; plot(t, sqrt(t), 'r-', 'LineWidth', 2); hold off;
end

%%

% Bound on how high a K we can expect to be able to demix; this is roughly 1 + round(Ls/6)

clc;
P = primes(100000)';
bound_on_K = zeros(size(Ls));
for iter_L = 1 : numel(Ls)
    L = Ls(iter_L);
    % make y to be the DFT of a real vector of length L, with prime numbers
    % for the Fourier magnitudes: this way, products of three distinct entries
    % (up to symmetries of real DFT's) are always distinct.
    if mod(L, 2) == 1 % odd
        p1 = P(1+(1:(L-1)/2));
        p2 = flipud(p1);
    else % even
        p1 = P(1+(1:L/2));
        p2 = flipud(p1(1:end-1));
    end
    y = [2 ; p1 ; p2] .* sign(fft(randn(L, 1)));
%     assert(norm(imag(ifft(y))) < 1e-14 * norm(y));
    B = (y*y').*circulant(y);
    u = abs([real(B(:)) ; imag(B(:))]);
    u = u(u ~= 0);
    u = uniquetol(u, 1e-13);
    T2 = ceil((L+1)/2); % number of distinct real numbers in power spectrum for a real signal of length L: changed Jan 29, 2018
    bound = floor(( 1+T2+numel(u) + 1 )/(L+1)); % Aug. 22, 2017: there was a bug here: the +1's are there because we need # >= KL + K-1 = K(L+1) - 1, so need K <= (#+1)/(L+1) -- the K-1 is because p is free in the simplex.
    bound_on_K(iter_L) = bound;
end

for sp = 1 : 4
    subplot(4, 1, sp);
    hold all;
    mask = bound_on_K <= max(Ks);
    plot(Ls(mask), bound_on_K(mask), 'r.', 'MarkerSize', 10);
    xlim([2, max(Ls)]);
    hold off;
end

%%
subplotsqueeze(gcf, 1.1);

%%
savefig('XP4.fig');
pdf_print_code(gcf, 'XP4.pdf');

