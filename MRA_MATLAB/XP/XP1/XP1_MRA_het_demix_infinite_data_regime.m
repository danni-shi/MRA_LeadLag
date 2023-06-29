
% June 30, 2017, NB
% Current code is noiseless regime rather than infinite data regime, in
% that there is no bias term added to the (perfect) mixed moments.

clear all; %#ok<CLALL>
close all;
clc;

%%

Ls = 1:1:100;
Ks = 1:10;
nrepeats = 30;
nmetrics = 3;
metric = zeros(nmetrics, length(Ls), length(Ks), nrepeats);
% Metric 1: relative estimation error
% Metric 2: objective value reached
% Metric 3: CPU time

opts = struct();
opts.maxiter = 200;
opts.tolgradnorm = 1e-10;
opts.tolcost = 1e-18;
    
fid = fopen('XP1_progress.txt', 'a');
origin = tic();
fprintf(fid, 'Starting: %s\r\n\r\n', datestr(now()));

for iter_K = 1 : length(Ks)
    
    K = Ks(iter_K);
        
    fprintf(fid, 'K = %3d, %s\r\nElapsed: %s [s]\r\n', K, datestr(now()), toc(origin));
    
    for iter_L = 1 : length(Ls)
        
        L = Ls(iter_L);
        
        if mod(iter_L, 10) == 0
            fprintf(fid, '\tL = %3d, %s\r\n', L, datestr(now()));
        end
        
        x_true = randn(L, K);
        
        parfor repeat = 1 : nrepeats
        
            % Solve from a new random initial guess.
            t = tic();
            [x_est, problem] = MRA_het_mixed_invariants(x_true, 0, K, [], opts);
            t = toc(t);

            % Evaluate quality of recovery, up to permutations and shifts.
            x_est = align_to_reference_het(x_est, x_true);
            relative_error = norm(x_est - x_true) / norm(x_true);
            metric(:, iter_L, iter_K, repeat) = [relative_error, getCost(problem, x_est), t];
            
        end
        
    end
    
    save XP1.mat;
    
end

fprintf(fid, 'Ending: %s\r\n\r\nElapsed: %s [s]\r\n', datestr(now()), toc(origin));
fclose(fid);

%%
save XP1.mat;

%%
load XP1;

figure(1);
clf;
metric1 = squeeze(metric(1, :, :, :));
metric2 = squeeze(metric(2, :, :, :));
metric3 = squeeze(metric(3, :, :, :));
% subplot(3, 1, 1);
% imagesc(Ls, Ks, log10(median(metric1, 3))');
% xlabel('L');
% ylabel('K');
% title('Median log10 of recovery error');
% set(gca, 'YDir', 'normal');
% colorbar;
% axis equal;
% axis tight;
subplot(3, 1, 1);
% imagesc(Ls, Ks, log10(min(metric2, [], 3))');
imagesc(Ls, Ks, mean(metric2 <= 1e-16, 3)');
% xlabel('L');
ylabel('K');
% title('Smallest objective value attained in log10');
title('Fraction of initializations leading to optimality');
set(gca, 'YDir', 'normal');
colorbar;
% axis equal;
pbaspect([5, 1, 1]);
axis tight;

subplot(3, 1, 2);
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
title('Largest relative estimation error among computed optima');
set(gca, 'YDir', 'normal');
colorbar;
% axis equal;
pbaspect([5, 1, 1]);
axis tight;

subplot(3, 1, 3);
imagesc(Ls, Ks, mean(log10(metric3), 3)');
xlabel('L');
ylabel('K');
% title('Smallest objective value attained in log10');
title('Average computation time (arithmetic mean, log_{10} scale)');
set(gca, 'YDir', 'normal');
colorbar;
% axis equal;
pbaspect([5, 1, 1]);
axis tight;

set(gcf, 'Color', 'w');

subplot(3, 1, 1); hold all; t = 1:100; plot(t, sqrt(t), 'r-', 'LineWidth', 2); hold off;
subplot(3, 1, 2); hold all; t = 1:100; plot(t, sqrt(t), 'r-', 'LineWidth', 2); hold off;
subplot(3, 1, 3); hold all; t = 1:100; plot(t, sqrt(t), 'r-', 'LineWidth', 2); hold off;

%%
subplotsqueeze(gcf, 1.15);

%%
savefig('XP1.fig');
pdf_print_code(gcf, 'XP1.pdf');


%% Bound on how high a K we can expect to be able to demix; this is roughly 1 + round(Ls/6)

%% June 30: avoid counting complex conjugates inside the upper triangular part (if that's even possible)
%  So, ignore sign, count distinct real numbers
return;
%%
clc;
P = primes(100000)';
Ls = 1:100;
bound_on_K = zeros(size(Ls));
uu = zeros(size(Ls));
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
    uu(iter_L) = numel(u);
    T2 = ceil((L+1)/2); % number of distinct real numbers in power spectrum for a real signal of length L: changed Jan 29, 2018
    bound = floor((1+T2+numel(u))/L);
    bound_on_K(iter_L) = bound;
    fprintf('L: %4d, nnz(u): %6d, bound on K: %3d\n', L, numel(u), bound);
end

% A rounded quadratic fits well (within 1) but not exactly.
% Could also fit a minimax polynomial to the data, try to get within 1/2
% error, so that rounding get the perfect answer? Or force exactly
% (1/6)Ls.^2 by subtracting it and fitting a degree one polynomial to the
% rest?
P = polyfit(Ls, uu, 2)
max(abs(round(polyval(P, Ls)) - uu))

%%
for sp = 1 : 3
    subplot(3, 1, sp);
    hold all;
    mask = bound_on_K <= max(Ks);
    plot(Ls(mask), bound_on_K(mask), 'r.', 'MarkerSize', 10);
    hold off;
    xlim([2, max(Ls)]);
end
