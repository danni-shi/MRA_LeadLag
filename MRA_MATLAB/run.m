function run(x_true, index, M, L, K, sigma, max_shift, nextrainits, save_data, folder_name, sigma_scale, train_ratio)
% run the entire data generation, signal recovery and recovery evaluation 
% process with different experiment settings.
    
    % Default values
    arguments
        x_true double;
        index double;
        M double {mustBePositive} = 1000;
        L double {mustBePositive} = 100;
        K double {mustBePositive} = 2;
        sigma double {mustBeNonnegative} = 1;
        max_shift double {mustBePositive} = 2;
        nextrainits double {mustBeNonnegative} = 0;
        save_data logical = true;
        folder_name string = sprintf('../data/data%i_%s',M,signal);
        sigma_scale double {mustBePositive} = 1;
        train_ratio double {mustBePositive} = 0.5;
        
    end

    
    
    % Ground truth mixing probabilities
    p_true = rand(K, 1);
    p_true = max(.8*(1/K), p_true);
    p_true = p_true / sum(p_true);
    
    % Number of measurements for each class
    Ms = round(p_true*M);
    
    % Generate the data
    [data, shifts, classes] = generate_observations_het(x_true, Ms, sigma, max_shift);
    % Split the data into train and test
    split_index = floor(train_ratio*L);
    data_train = data(1:split_index,:);
    data_test = data(split_index+1:end,:);
    x_true_train = x_true(1:split_index,:);
    x_true_test = x_true(split_index+1:end,:);
    index_train = index(1:split_index,:);
    index_test = index(split_index+1:end,:);
    %% Optimization
    
    opts = struct();
    opts.maxiter = 200;
    opts.tolgradnorm = 1e-7;
    opts.tolcost = 1e-18;
    opts.verbosity = 1;
    
    % initial point
    X0 = zeros(L,1);
    p0 = ones(K, 1) / K;
    
    [x_est, p_est, problem] = MRA_het_mixed_invariants_free_p(data_train, sigma_scale*sigma, K, [], [], opts, [], nextrainits);
    
    %% Evaluate quality of recovery, up to permutations and shifts.
    
    [x_est, E, perm] = align_to_reference_het(x_est, x_true_train);
    p_est = p_est(perm);
    rel_error_X = zeros(K,1);
    tv_error_p = norm(p_est - p_true, 1) / 2;
    
    for k = 1 : K
        rel_error_X(k) = norm(x_est(:, k) - x_true_train(:, k)) / norm(x_true_train(:, k));
    end
    
    if save_data
        % folder_name = sprintf('../data/data%i_%s',M,signal);
        file_specs = sprintf('noise%.3g_shift%.3g_class%i',sigma, max_shift, K);
        save(sprintf('%s/observations_%s.mat', folder_name, file_specs), 'data_train','data_test', 'shifts', 'classes','index_train','index_test');
        save(sprintf('%s/results_%s.mat', folder_name, file_specs), 'x_true_train', 'x_true_test', 'x_est', 'p_est', 'rel_error_X', 'tv_error_p')
    else              
        %% Plot
        d1 = floor(sqrt(K));
        d2 = ceil(K/d1);
        for k = 1 : K
            subplot(d1, d2, k);
            plot(1:L, x_true(:, k), '.-', 1:L, x_est(:, k), 'o-');
            legend('True signal', 'Estimate');
            title(sprintf('True weight: %.3g; estimated: %.3g, Relative Error: %.3g, Lag: %i', p_true(k), p_est(k), rel_error_X(k)));
        end
    end
    

