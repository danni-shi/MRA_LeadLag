clear all; %#ok<CLALL>
close all;
clc;

rng(42);
save_data = true;

%% Problem setup
M = 500;
L = 100; % signal length
signal = 'pvCLCL';
max_shift = 2; % shift set to -1 to enable cyclic shifts
nextrainits = 2;
train_ratio = 0.5;
num_rounds = 4;
sigma_scale = 1;
% K = 3;  % number of different signals to estimate (heterogeneity)
% sigma = 0.5;
% sigma_scale = 1.5;
% run(M, L, K, sigma, max_shift, signal, nextrainits, save_data,[],sigma_scale);

if save_data == true    
      % create folder to store the data
    folder_name = sprintf('../data/data%i_shift%.3g_%s_init%i_set1',M,max_shift,signal,nextrainits);
    if ~exist(folder_name, 'dir')
%         rmdir(folder_name,'s');
        mkdir(folder_name);
    end
    %mkdir(folder_name);
end


for round = 1:num_rounds
    if save_data == true    
       % create folder to store the data
       subfolder_name = strcat(folder_name,sprintf('/%i',round));
       mkdir(subfolder_name);
    end

    for K = 2:2
        % generate observations at different noise level for the same
        % latent signal of shape LxK
        [x_true, SPY] = generate_signal(K,L,signal);
        for sigma = 0.1:0.1:2.0
            fprintf('K = %i, sigma = %.3g \n', K, sigma)
            run(x_true, SPY, M, L, K, sigma, max_shift, nextrainits, save_data, subfolder_name,sigma_scale,train_ratio)
        end
    end
end

% generate data at different noise level and different number of classes
%  diary log500_50_init3_nonPeriodicSine.txt



% diary off

% L = 50; % signal length
% K = 1;  % number of different signals to estimate (heterogeneity)
% signal = 'gaussian';
% sigma = 1;
% max_shift = 0.1; % shift set to -1 to enable cyclic shifts
% 
% run(L, K, sigma, max_shift, signal, save_data);
