% =========================================================================
% Advanced MATLAB Simulation for MIMO Downlink Precoding Performance
% =========================================================================
% Author: Gemini
% Date: 2025-04-23
% Version: 3.7 (Per-User Hybrid ZF/MRT Precoding Optimization)
% Description: Implements a hybrid ZF/MRT scheme where each user has an
%              independent weight lambda_k, optimized via multi-dimensional
%              gradient ascent with adaptive learning rates.
%              Builds upon v3.3 structure.
% =========================================================================

clear;
clc;
close all;

%% Simulation Parameters Structure
params = struct();

% --- System Parameters ---
params.Nt = 155;          % Number of transmit antennas at BS (Reduced for faster BER sim)
params.K = 150;           % Number of single-antenna users (Reduced for faster BER sim)
params.total_power = 1; % Total transmit power constraint at the BS

% --- Channel Model Parameters ---
params.channel_model = 'Rayleigh'; % 'Rayleigh' or 'Rician'
params.rician_K_factor = 10;     % Rician K-factor (linear scale), only for 'Rician'

% --- Modulation and BER Parameters ---
params.mod_order = 4;           % Modulation order (e.g., 4 for QPSK, 16 for 16-QAM)
params.bits_per_symbol = log2(params.mod_order);
params.num_bits_per_symbol_target = 1e4; % Target number of bits per user per SNR point for BER
                                        % Increase for smoother BER curves, decrease for speed.
params.num_symbols_per_run = ceil(params.num_bits_per_symbol_target / params.bits_per_symbol);

% --- Simulation Control ---
params.SNR_dB_vec = 20:1:20; % SNR range in dB
params.num_monte_carlo = 10; % Number of Monte Carlo channel realizations.  蒙特卡罗循环次数
                              % BER averaging happens *inside* MC loop over symbols.

% --- Precoding Scheme Selection ---
% MODIFIED: Added 'Hybrid_ZF_MRT'
params.precoding_schemes = {'ZF', 'MMSE', 'MRT', 'Hybrid_ZF_MRT_03'};
%'Hybrid_ZF_MRT'是总体归一化,'Hybrid_ZF_MRT_02'是各用户分别归一化，Hybrid_ZF_MRT_03'是文章中的方法

% --- Hybrid Precoding Optimization Parameters ---
params.hybrid.lambda_init = 0.5;      % Initial weighting factor lambda
params.hybrid.learning_rate_init = 0.1; % Initial learning rate for gradient ascent
params.hybrid.lr_reduction_factor = 10; % Factor to reduce learning rate when gradient flips
params.hybrid.max_iterations = 1000;    % Max iterations for lambda optimization per channel
params.hybrid.gradient_delta = 1e-4;  % Small delta for numerical gradient calculation
params.hybrid.convergence_threshold = 1e-6; % Threshold for lambda change to declare convergence

params.hybrid.grad_clip=1;%梯度截断阈值 
% --- Visualization Control ---
% Select metric to plot: 
params.plot_metric = 'SumRate';%'SumRate', 'BER', 'MultiplexingGain', 'DiversityGain',SpectralEfficiency

params.plot_sumrate = 'ture';
params.plot_ber = 'false';
params.plot_se = 'false';

%% Initialization (Adapts automatically to new scheme)
num_snr_points = length(params.SNR_dB_vec);
num_schemes = length(params.precoding_schemes);
snr_linear_vec = 10.^(params.SNR_dB_vec / 10); % Linear SNR values

%% Use a structure to store results
results = struct();
fprintf('Initializing results structure for: %s\n', strjoin(params.precoding_schemes, ', '));
for i = 1:num_schemes
    scheme_name = params.precoding_schemes{i};
    results.(scheme_name).sum_rate = zeros(1, num_snr_points);
    results.(scheme_name).ber = zeros(1, num_snr_points);
    results.(scheme_name).total_bits_simulated = zeros(1, num_snr_points);
    results.(scheme_name).total_errors_counted = zeros(1, num_snr_points);
    % Optional: Store optimized lambda values for analysis
    if strcmpi(scheme_name, 'Hybrid_ZF_MRT')|| strcmpi(scheme_name, 'Hybrid_ZF_MRT_02') || strcmpi(scheme_name, 'Hybrid_ZF_MRT_03')
       results.(scheme_name).optimized_lambda = zeros(params.K,params.num_monte_carlo, num_snr_points);
    end
end
fprintf('Initialization complete.\n');


% --- [Simulation Loop Structure Remains Similar] ---
% Inside the main SNR loop and Monte Carlo loop:
% When scheme_idx corresponds to 'Hybrid_ZF_MRT', the apply_precoding function
% will perform the optimization and return the optimized W.
% The calculate_performance_metrics function is called *after* apply_precoding
% returns the final W for the hybrid scheme, just like for other schemes.

% --- [generate_channel_model remains the same] ---
% --- [calculate_performance_metrics remains the same] ---
% --- [custom_qammod and custom_qamdemod remain the same] ---
% --- [plot_results remains the same, will plot the new scheme automatically] ---

%% Simulation Loop  主循环
fprintf('Starting Advanced MIMO precoding simulation...\n');
fprintf('Channel Model: %s\n', params.channel_model);
fprintf('Modulation Order: %d-QAM\n', params.mod_order);
fprintf('Precoding Schemes: %s\n', strjoin(params.precoding_schemes, ', '));
tic; % Start timer

for snr_idx = 1:num_snr_points   %不同SNR
    snr_db = params.SNR_dB_vec(snr_idx);
    snr_linear = snr_linear_vec(snr_idx);
    noise_variance = params.total_power / snr_linear;

    fprintf('  Simulating SNR = %d dB...\n', snr_db);

    % Accumulators for BER across Monte Carlo runs for this SNR point 每次SNR重置结果参数
    bits_simulated_snr = zeros(1, num_schemes);
    errors_counted_snr = zeros(1, num_schemes);
    sum_rate_acc_snr = zeros(1, num_schemes); % Accumulator for sum rate
    actual_tx_power_acc_snr = zeros(1, num_schemes); % 新增: 累加实际发射功率
    
    % Temporary storage for optimized lambda values within this SNR point
    lambda_opt_storage = zeros(params.num_monte_carlo, 1); % Only needed if averaging lambda
    for mc_run = 1:params.num_monte_carlo  %蒙特卡洛循环十次，进行十次结果累加并求均值
        % Display progress lightly
        % if mod(mc_run, params.num_monte_carlo/100) == 0
        %    fprintf('    MC run %d/%d\n', mc_run, params.num_monte_carlo);
        % end
        fprintf('    MC run %d/%d\n', mc_run, params.num_monte_carlo);
        % 1. Generate Channel Matrix
        H = generate_channel_model(params); % Size: K x Nt

        % 2. Apply Precoding and Calculate Performance for each scheme
        for scheme_idx = 1:num_schemes
            precoder_type = params.precoding_schemes{scheme_idx};

            % Apply the selected precoding (includes normalization)获得最终的W和平衡参数
             [W, lambda_opt] = apply_precoding(H, noise_variance, precoder_type, params);

            if (strcmpi(precoder_type, 'Hybrid_PerUser_Opt') || strcmpi(precoder_type, 'Hybrid_PerUser_Norm')) && ~all(isnan(lambda_opt_vec))
                 results.(precoder_type).optimized_lambda(:, mc_run, snr_idx) = lambda_opt_vec;
            end

            % --- Performance Calculation ---
            % 新增: 计算并累加实际发射功率
            if ~isempty(W) && all(isfinite(W(:)))
                actual_power_W = trace(W*W');
                actual_tx_power_acc_snr(scheme_idx) = actual_tx_power_acc_snr(scheme_idx) + actual_power_W;
            else
                % 如果W无效，则功率视为0或NaN，避免影响累加
                actual_tx_power_acc_snr(scheme_idx) = actual_tx_power_acc_snr(scheme_idx) + NaN;
            end
            
           if(strcmpi(params.plot_ber, 'ture')) % --- Generate Data Symbols for BER ---  计算BER需要生成数据符号
            % Generate random bits for all users  随机生成数据比特流
            tx_bits = randi([0 1], params.K * params.bits_per_symbol, params.num_symbols_per_run);
            % Modulate bits to symbols for each user stream
            % Note: calculate_performance_metrics will handle modulation internally now

            % tx_symbols_vec = reshape(tx_bits, params.K, params.bits_per_symbol * params.num_symbols_per_run);
            % tx_symbols = zeros(params.K, params.num_symbols_per_run);%初始化发射端数据符号流

          
            metrics = calculate_ber_metrics(H, W, noise_variance, tx_bits, params);
            errors_counted_snr(scheme_idx) = errors_counted_snr(scheme_idx) + metrics.error_count;
            bits_simulated_snr(scheme_idx) = bits_simulated_snr(scheme_idx) + metrics.bits_processed;
           end 
            
           if(strcmpi(params.plot_sumrate, 'ture')) 
            metrics = calculate_sumrate_metrics(H, W, noise_variance, params);
            % Accumulate results for averaging，累加求平均值
            sum_rate_acc_snr(scheme_idx) = sum_rate_acc_snr(scheme_idx) + metrics.sum_rate;
           
           end
        end % End precoding scheme loop
    end % End Monte Carlo loop

    % Average results over Monte Carlo runs and store
    for scheme_idx = 1:num_schemes
        scheme_name = params.precoding_schemes{scheme_idx};
        results.(scheme_name).sum_rate(snr_idx) = sum_rate_acc_snr(scheme_idx) / params.num_monte_carlo;

        % Calculate BER for this SNR point
        if bits_simulated_snr(scheme_idx) > 0
            results.(scheme_name).ber(snr_idx) = errors_counted_snr(scheme_idx) / bits_simulated_snr(scheme_idx);
        else
            results.(scheme_name).ber(snr_idx) = NaN; % Avoid division by zero if no bits were simulated
        end
        results.(scheme_name).total_bits_simulated(snr_idx) = bits_simulated_snr(scheme_idx);
        results.(scheme_name).total_errors_counted(snr_idx) = errors_counted_snr(scheme_idx);
        results.(scheme_name).total_tx_power(snr_idx) = actual_tx_power_acc_snr(scheme_idx) / params.num_monte_carlo; % 新增: 存储平均实际发射功率
        % Display BER results for verification
        if(strcmpi(params.plot_ber, 'ture'))
         fprintf('    %s: Errors=%d, Bits=%d, BER=%.2e\n', ...
                 scheme_name, errors_counted_snr(scheme_idx), ...
                 bits_simulated_snr(scheme_idx), results.(scheme_name).ber(snr_idx));
        end
        if(strcmpi(params.plot_sumrate, 'ture'))
         fprintf('    %s: Errors=%d, Bits=%d, BER=%.2e\n', ...
                 scheme_name,  sum_rate_acc_snr(scheme_idx) / params.num_monte_carlo);
        end

    end

end % End SNR loop

toc; % Stop timer
fprintf('Simulation finished.\n');

%% Results Visualization
plot_results(results, params);


%% Helper Functions (generate_channel_model and apply_precoding are unchanged from v2)

% =========================================================================
% Interface Function to Generate Channel Matrix based on Model Type
% (Unchanged from v2 - Keep for completeness)
% =========================================================================
function H = generate_channel_model(params)
    % Generates a K x Nt MIMO channel matrix based on the specified model.
    switch lower(params.channel_model)
        case 'rayleigh'
            H = (randn(params.K, params.Nt) + 1i * randn(params.K, params.Nt)) / sqrt(2);
        case 'rician'
            K_factor = params.rician_K_factor;
            H_nlos = (randn(params.K, params.Nt) + 1i * randn(params.K, params.Nt)) / sqrt(2);
            % Simple fixed LoS component (random per realization)
            H_los = exp(1i * 2 * pi * rand(params.K, params.Nt)); % Random phase LoS
            H = sqrt(K_factor / (K_factor + 1)) * H_los + sqrt(1 / (K_factor + 1)) * H_nlos;
        otherwise
            error('Unknown channel model: %s', params.channel_model);
    end
end

% =========================================================================
% Internal Helper Function: Normalize Precoding Matrix
% =========================================================================
function W_norm = normalize_precoder(W_un, P_total)
    % Normalizes the precoding matrix W_un to satisfy trace(W*W') = P_total.
    norm_W_sq = trace(W_un * W_un');
    if norm_W_sq > 1e-10 % Check if norm is reasonably non-zero
        power_scaling_factor = sqrt(P_total / norm_W_sq);
        W_norm = power_scaling_factor * W_un;
    else
        % If norm is zero or very small, return zero matrix
        [Nt, K] = size(W_un);
        W_norm = zeros(Nt, K);
    end
    % Final check for safety
    if any(isnan(W_norm(:))) || any(isinf(W_norm(:)))
         [Nt, K] = size(W_un);
         W_norm = zeros(Nt, K);
    end
end

% =========================================================================
% Internal Helper Function: Calculate Sum Rate (for gradient ascent)
% =========================================================================
function sum_rate = calculate_sum_rate_internal(H, W, noise_variance, K)
    % Calculates sum rate given H, normalized W, noise_variance.
    % Simplified version of calculate_performance_metrics, only computes sum rate.
    sum_rate = 0;
    if any(isnan(W(:))) || any(isinf(W(:))) || isempty(W) || norm(W) < 1e-9
        % If W is invalid (e.g., due to normalization issues), rate is 0
        sum_rate = 0;
        return;
    end

    H_eff = H * W; % Effective channel (K x K)
    for k = 1:K
        signal_power = abs(H_eff(k, k))^2;
        interference_power = norm(H_eff(k, :))^2 - signal_power;
        interference_power = max(0, interference_power); % Ensure non-negative
        sinr_k = signal_power / (interference_power + noise_variance);
        if isnan(sinr_k) || isinf(sinr_k) || sinr_k < 0 || signal_power < 1e-15
             rate_k = 0; % Handle invalid SINR or near-zero signal
        else
            rate_k = log2(1 + sinr_k);
        end
        sum_rate = sum_rate + rate_k;
    end
     % Ensure sum_rate is not NaN/Inf
     if isnan(sum_rate) || isinf(sum_rate)
         sum_rate = 0;
     end
end
% =========================================================================
% NEW Helper Function: Optimize ("Train") Lambda for Per-User Hybrid Scheme
% =========================================================================
function [lambda_opt_vec, W_mrt_un_cols, W_zf_un_cols, zf_col_possible] = optimize_hybrid_per_user_lambda(H, noise_variance, params)
    % Performs multi-dimensional gradient ascent to find the optimal lambda vector.
    % This function effectively "trains" the lambda parameters for the current channel realization.
    % It uses numerical gradients to maximize the sum rate after overall power normalization.
    % Inputs: H, noise_variance, params
    % Outputs:
    %   lambda_opt_vec: Optimized ("Trained") Kx1 lambda vector
    %   W_mrt_un_cols: Unnormalized MRT column vectors (Nt x K)
    %   W_zf_un_cols: Unnormalized ZF column vectors (Nt x K)
    %   zf_col_possible: Logical vector (1 x K) indicating if ZF basis was valid

    Nt = params.Nt;
    K = params.K;

    % 1. Calculate Base Precoding Matrices (Columns)
    W_mrt_un_cols = H'; % MRT columns
    W_zf_un_cols = zeros(Nt, K);
    zf_col_possible = true(1, K); % Track if ZF is possible
    try
        HH_H = H * H'; % Calculate once
        if cond(HH_H) > 1e10
           warning('HybridPerUserOpt Training: H*H^H ill-conditioned (cond=%e). ZF part might be unstable.', cond(HH_H));
           % Mark ZF as potentially unstable, but try calculating inverse
        end
        HH_H_inv = inv(HH_H);
        W_zf_un_cols = H' * HH_H_inv; % Calculate all ZF columns
    catch ME
         warning('HybridPerUserOpt Training: Could not compute inv(H*H^H) for ZF part (%s). Disabling ZF.', ME.identifier);
         zf_col_possible(:) = false; % Disable ZF for all columns if matrix inversion fails
         W_zf_un_cols = zeros(Nt, K); % Ensure ZF columns are zero if unusable
    end

    % Initialization for Gradient Ascent (Lambda Training)
    lambda_vec_k = ones(K, 1) * params.hybrid.lambda_init; % Initial lambda values
    eta_vec = ones(K, 1) * params.hybrid.learning_rate_init; % Initial learning rates
    grad_vec_prev = zeros(K, 1); % Previous gradient vector

    % --- Optimization Loop (Training Iterations) ---
    for iter = 1:params.hybrid.max_iterations
        lambda_vec_prev = lambda_vec_k;

        % Calculate base sum rate for current lambda_vec_k (using overall normalization)
        W_un_k = zeros(Nt, K);
        for i = 1:K
            if zf_col_possible(i)
               W_un_k(:, i) = lambda_vec_k(i) * W_mrt_un_cols(:, i) + (1 - lambda_vec_k(i)) * W_zf_un_cols(:, i);
            else
                W_un_k(:, i) = W_mrt_un_cols(:, i);
                lambda_vec_k(i) = 1.0; % Ensure lambda reflects MRT usage if ZF impossible
            end
        end
        W_k = normalize_precoder(W_un_k, params.total_power); % Apply overall normalization
        rate_base = calculate_sum_rate_internal(H, W_k, noise_variance, K); % Calculate sum rate

        % Calculate gradient vector numerically (Forward Difference)
        % This estimates d(SumRate)/d(lambda_i) considering the overall normalization effect
        grad_vec_k = zeros(K, 1);
        delta = params.hybrid.gradient_delta;
        for i = 1:K % Iterate through each lambda_i to estimate partial derivative
            lambda_vec_perturbed = lambda_vec_k;
            original_lambda_i = lambda_vec_k(i);
            lambda_vec_perturbed(i) = min(1, original_lambda_i + delta); % Perturb lambda_i

            % Skip gradient calculation if lambda cannot change or ZF is impossible for this user
            if abs(lambda_vec_perturbed(i) - original_lambda_i) < 1e-9 || ~zf_col_possible(i)
                grad_vec_k(i) = 0;
                continue;
            end

            % Construct W_un with perturbed lambda_i
            W_un_perturbed = zeros(Nt, K);
             for j = 1:K % Reconstruct the whole unnormalized matrix
                 if zf_col_possible(j)
                     W_un_perturbed(:, j) = lambda_vec_perturbed(j) * W_mrt_un_cols(:, j) + (1 - lambda_vec_perturbed(j)) * W_zf_un_cols(:, j);
                 else
                     W_un_perturbed(:, j) = W_mrt_un_cols(:, j);
                 end
             end
            W_perturbed = normalize_precoder(W_un_perturbed, params.total_power); % Normalize the perturbed matrix
            rate_perturbed = calculate_sum_rate_internal(H, W_perturbed, noise_variance, K); % Calculate new sum rate

            % Estimate gradient component
            grad_vec_k(i) = (rate_perturbed - rate_base) / (lambda_vec_perturbed(i) - original_lambda_i);
        end

        % Update lambda vector and adapt learning rates based on gradient
        lambda_changed = false; % Flag to check if any lambda actually changed
        for i = 1:K
            if ~zf_col_possible(i) % Keep lambda fixed at 1 if ZF impossible
                lambda_vec_k(i) = 1.0;
                continue;
            end

            % Adapt learning rate eta_i if gradient sign flips
            if iter > 1 && grad_vec_k(i) * grad_vec_prev(i) < -1e-12 % Check for sign flip
                eta_vec(i) = eta_vec(i) / params.hybrid.lr_reduction_factor;
            end

            % Update lambda_i using gradient ascent step
          
            new_lambda_i = lambda_vec_k(i) + eta_vec(i) * grad_vec_k(i);
            new_lambda_i = max(0, min(1, new_lambda_i)); % Project lambda_i back to [0, 1]

            if abs(new_lambda_i - lambda_vec_k(i)) > 1e-9 % Check if change is significant
                lambda_changed = true;
            end
            lambda_vec_k(i) = new_lambda_i;
        end

        grad_vec_prev = grad_vec_k; % Store current gradient for next iteration's check

        % Check convergence conditions
        if norm(lambda_vec_k - lambda_vec_prev) < params.hybrid.convergence_threshold || ~lambda_changed
            % fprintf('Lambda training converged at iteration %d\n', iter);
            break; % Exit training loop if converged or no change
        end

    end % End gradient ascent loop (Training Complete)

    lambda_opt_vec = lambda_vec_k; % Assign final ("trained") lambda vector
end


% =========================================================================
% NEW Helper Function: Construct Hybrid Precoding Matrix from Optimized Lambdas
% =========================================================================
function W = apply_hybrid_per_user_precoding(lambda_opt_vec, W_mrt_un_cols, W_zf_un_cols, zf_col_possible, params)
    % Constructs the final normalized hybrid precoding matrix using optimized ("trained") lambdas.
    % Inputs:
    %   lambda_opt_vec: Optimized Kx1 lambda vector
    %   W_mrt_un_cols: Unnormalized MRT column vectors (Nt x K)
    %   W_zf_un_cols: Unnormalized ZF column vectors (Nt x K)
    %   zf_col_possible: Logical vector (1 x K) indicating if ZF basis was valid
    %   params: System parameters structure (for K, Nt, total_power)
    % Output:
    %   W: Final normalized precoding matrix (Nt x K)

    Nt = params.Nt;
    K = params.K;

    % Construct final unnormalized precoder using optimized lambda vector
    W_unnormalized = zeros(Nt, K);
    for i = 1:K
         if zf_col_possible(i)
             W_unnormalized(:, i) = lambda_opt_vec(i) * W_mrt_un_cols(:, i) + (1 - lambda_opt_vec(i)) * W_zf_un_cols(:, i);
         else
             W_unnormalized(:, i) = W_mrt_un_cols(:, i); % Use MRT if ZF failed
         end
    end

    % Final normalization using the helper function
    W = normalize_precoder(W_unnormalized, params.total_power);
end
% =========================================================================
% NEW Helper Function: Optimize Lambda for Per-User NORM Hybrid Scheme
% =========================================================================
function [lambda_opt_vec, W_mrt_un_cols, W_zf_un_cols, zf_col_possible] = optimize_hybrid_per_user_norm_lambda(H, noise_variance, params)
    % Performs multi-dimensional gradient ascent to find the optimal lambda vector
    % for the PER-USER NORMALIZED hybrid scheme.
    % Numerical gradients are based on sum rate calculated with per-user normalized W.
    % Inputs: H, noise_variance, params
    % Outputs:
    %   lambda_opt_vec: Optimized Kx1 lambda vector
    %   W_mrt_un_cols: Unnormalized MRT column vectors (Nt x K)
    %   W_zf_un_cols: Unnormalized ZF column vectors (Nt x K)
    %   zf_col_possible: Logical vector (1 x K) indicating if ZF basis was valid

    Nt = params.Nt;
    K = params.K;

    % 1. Calculate Base Precoding Matrices (Columns) - Same as other hybrid
    W_mrt_un_cols = H';
    W_zf_un_cols = zeros(Nt, K);
    zf_col_possible = true(1, K);
    try
        HH_H = H * H';
        if cond(HH_H) > 1e10, warning('HybridPerUserNorm Training: H*H^H ill-conditioned.'); end
        HH_H_inv = inv(HH_H);
        W_zf_un_cols = H' * HH_H_inv;
    catch ME
        warning('HybridPerUserNorm Training: Could not compute inv(H*H^H) for ZF part (%s). Disabling ZF.', ME.identifier);
        zf_col_possible(:) = false;
        W_zf_un_cols = zeros(Nt, K);
    end

    % Initialization for Gradient Ascent
    lambda_vec_k = ones(K, 1) * params.hybrid.lambda_init; % Use specific params
    eta_vec = ones(K, 1) * params.hybrid.learning_rate_init;
    grad_vec_prev = zeros(K, 1);

    % --- Optimization Loop (Training Iterations) ---
    for iter = 1:params.hybrid.max_iterations
        lambda_vec_prev = lambda_vec_k;

        % Calculate base sum rate for current lambda_vec_k using PER-USER normalization
        W_k = apply_hybrid_per_user_norm_precoding(lambda_vec_k, W_mrt_un_cols, W_zf_un_cols, zf_col_possible, params);
        rate_base = calculate_sum_rate_internal(H, W_k, noise_variance, K); % Rate calculation uses the final W

        % Calculate gradient vector numerically (Forward Difference)
        grad_vec_k = zeros(K, 1);
        delta = params.hybrid.gradient_delta;
        for i = 1:K
            lambda_vec_perturbed = lambda_vec_k;
            original_lambda_i = lambda_vec_k(i);
            lambda_vec_perturbed(i) = min(1, original_lambda_i + delta);

            if abs(lambda_vec_perturbed(i) - original_lambda_i) < 1e-9 || ~zf_col_possible(i)
                grad_vec_k(i) = 0;
                continue;
            end

            % Construct W with perturbed lambda_i using PER-USER normalization
            W_perturbed = apply_hybrid_per_user_norm_precoding(lambda_vec_perturbed, W_mrt_un_cols, W_zf_un_cols, zf_col_possible, params);
            rate_perturbed = calculate_sum_rate_internal(H, W_perturbed, noise_variance, K);

            grad_vec_k(i) = (rate_perturbed - rate_base) / (lambda_vec_perturbed(i) - original_lambda_i);
        end

        % Update lambda vector and adapt learning rates
        lambda_changed = false;
        for i = 1:K
            if ~zf_col_possible(i)
                lambda_vec_k(i) = 1.0;
                continue;
            end
            if iter > 1 && grad_vec_k(i) * grad_vec_prev(i) < -1e-12
                eta_vec(i) = eta_vec(i) / params.hybrid.lr_reduction_factor;
            end
            new_lambda_i = lambda_vec_k(i) + eta_vec(i) * grad_vec_k(i);
            new_lambda_i = max(0, min(1, new_lambda_i));
            if abs(new_lambda_i - lambda_vec_k(i)) > 1e-9, lambda_changed = true; end
            lambda_vec_k(i) = new_lambda_i;
        end

        grad_vec_prev = grad_vec_k;

        % Check convergence
        if norm(lambda_vec_k - lambda_vec_prev) < params.hybrid.convergence_threshold || ~lambda_changed
            break;
        end
    end % End gradient ascent loop

    lambda_opt_vec = lambda_vec_k;
end


% =========================================================================
% NEW Helper Function: Construct Hybrid Precoding Matrix (Per-User Norm)
% =========================================================================
function W = apply_hybrid_per_user_norm_precoding(lambda_opt_vec, W_mrt_un_cols, W_zf_un_cols, zf_col_possible, params)
    % Constructs the final hybrid precoding matrix using PER-USER normalization.
    % Inputs: lambda_opt_vec, W_mrt_un_cols, W_zf_un_cols, zf_col_possible, params
    % Output: W: Final precoding matrix (Nt x K), likely Tr(W*W') = K

    Nt = params.Nt;
    K = params.K;
    W = zeros(Nt, K); % Initialize final matrix

    for i = 1:K
        % Construct the unnormalized vector for user i
        if zf_col_possible(i)
            w_un_i = lambda_opt_vec(i) * W_mrt_un_cols(:, i) + (1 - lambda_opt_vec(i)) * W_zf_un_cols(:, i);
        else
            w_un_i = W_mrt_un_cols(:, i); % Use MRT if ZF failed
        end

        % Normalize this vector independently
        norm_wi = norm(w_un_i, 'fro'); % Use Frobenius norm for vectors
        if norm_wi > 1e-10
            W(:, i) = w_un_i / norm_wi; % Normalize column i to have unit norm
        else
            W(:, i) = zeros(Nt, 1); % Assign zero vector if norm is too small
        end
         % Safety check for NaN/Inf in the column
        if any(isnan(W(:, i))) || any(isinf(W(:, i)))
             W(:, i) = zeros(Nt, 1);
        end
    end
end

% =========================================================================
% NEW Main Function for User's Hybrid Scheme (Analytic Gradient, Ind. Norm)
% Adapted from user's precoding_module
% =========================================================================
function [lambda_opt_vec, W_opt] = run_hybrid_analytic_indnorm(H, sigma2, params)
    % Wrapper for the user-provided hybrid scheme.
    % H: Channel matrix
    % sigma2: Noise variance (fixed to 1.0 in this simulation setup)
    % params Main simulation params structure

    % Extract specific parameters for this optimization method
    
    K = params.K;
    Nt = params.Nt; % Framework uses Nt, user code used N

    % Initialize parameters as in user's precoding_module
    eta_vec = ones(K, 1) * params.hybrid.learning_rate_init; % Initial learning rates
    lambda = params.hybrid.lambda_init* ones(K, 1); % Initial lambda
    prev_grad = zeros(K, 1);
    prev_rate = -Inf; % Initialize to a very small number

    % Calculate initial unnormalized precoding matrices
    W_MRT_un = H'; % MRT precoding (Nt x K)
    W_ZF_un = H' * pinv(H * H'); % ZF precoding (Nt x K)

    % Iterative optimization loop
    for iter = 1:params.hybrid.max_iterations
        % Generate current precoding matrix W with per-user normalization
        % This W will have Tr(W*W') = K
        W_current = analytic_param_precoding(lambda, W_MRT_un, W_ZF_un);

        % Calculate gradient using user's analytic method
        % Note: calc_gradient_analytic needs W_current (normalized per user)
        % and also W_MRT_un, W_ZF_un (unnormalized columns for df_ki_dlambda_analytic)
        % 计算得到对每个用户的梯度
        grad = calc_gradient_analytic(H, W_current, W_MRT_un, W_ZF_un, lambda, sigma2, params);

        % Gradient clipping
        grad = max(-params.hybrid.grad_clip, min(params.hybrid.grad_clip, grad));

        % Calculate current sum rate for learning rate adjustment
        [C_sum_current, ~] = analytic_calc_rate(H, W_current, sigma2);

        % Dynamic learning rate adjustment
        if iter > 1
            sign_changes = (grad .* prev_grad) < -1e-12; % Check for actual sign flip
            rate_decrease = C_sum_current < prev_rate;
            for i = 1:K
                if sign_changes(i) || rate_decrease % User's original condition
                    eta_vec(i) = eta_vec(i) / params.hybrid.lr_reduction_factor;
                    % fprintf('User %d Analytic: LR adjusted to %.4e at iter %d\n', i, alpha(i), iter);
                end
            end
        end
        prev_grad = grad;
        prev_rate = C_sum_current;

        % Update lambda and project to [0,1]
        lambda_new = lambda + eta_vec .* grad; % Gradient ASCENT
        lambda_new = max(0, min(1, lambda_new));

        % Check convergence
        if norm(lambda_new - lambda) < params.hybrid.convergence_threshold
            % fprintf('Hybrid_Analytic_IndNorm converged at iteration %d.\n', iter);
            break;
        end
        lambda = lambda_new;

        % Optional: Output current iteration info (can be verbose)
        % fprintf('Iter: %d (Analytic), Sum Rate: %.4f\n', iter, C_sum_current);
    end

    lambda_opt_vec = lambda;
    W_opt = analytic_param_precoding(lambda_opt_vec, W_MRT_un, W_ZF_un); % Final W
end

% --- Helper functions adapted from user's code ---
function W_norm_pu = analytic_param_precoding(lambda_vec, W_MRT_un, W_ZF_un)
    % lambda_vec: Kx1 vector of lambdas
    % W_MRT_un: Nt x K unnormalized MRT matrix
    % W_ZF_un: Nt x K unnormalized ZF matrix
    % Output: W_norm_pu: Nt x K matrix, each column normalized to 1
    [Nt, K] = size(W_MRT_un);
    W_norm_pu = zeros(Nt, K);
    for i = 1:K
        w_un_i = lambda_vec(i) * W_MRT_un(:, i) + (1 - lambda_vec(i)) * W_ZF_un(:, i);
        norm_wi = norm(w_un_i, 2);
        if norm_wi > 1e-10
            W_norm_pu(:, i) = w_un_i / norm_wi;
        else
            W_norm_pu(:, i) = zeros(Nt, 1); % Or some other handling for zero norm
        end
    end
end

function grad_vec = calc_gradient_analytic(H_chan, W_curr_pu_norm, W_MRT_un_cols, W_ZF_un_cols, lambda_vec, sigma2_noise, params)
    % H_chan: K x Nt channel matrix
    % W_curr_pu_norm: Nt x K current precoding matrix (columns independently normalized)
    % W_MRT_un_cols, W_ZF_un_cols: Nt x K unnormalized base precoders
    % lambda_vec: Kx1 current lambdas
    % sigma2_noise: noise variance
    K_users = params.K;
    grad_vec = zeros(K_users, 1);

    % Pre-calculate all f_ki = |h_k * w_i(lambda_i)|^2 where w_i is from W_curr_pu_norm
    F_ki = zeros(K_users, K_users);
    for k_rx = 1:K_users
        for i_tx = 1:K_users
            F_ki(k_rx, i_tx) = abs(H_chan(k_rx, :) * W_curr_pu_norm(:, i_tx))^2;
        end
    end

    % Pre-calculate all df_ki/dlambda_i (derivative of |h_k * w_i(lambda_i)|^2 w.r.t lambda_i)
    % where w_i(lambda_i) is the per-user normalized vector
    dF_dlambda_i = zeros(K_users, K_users); % dF_dlambda_i(k,i) means df_ki / dlambda_i
    for k_rx = 1:K_users
        for i_tx = 1:K_users % The derivative is w.r.t lambda_i (i.e., lambda_vec(i_tx))
            dF_dlambda_i(k_rx, i_tx) = df_ki_dlambda_analytic(lambda_vec(i_tx), H_chan(k_rx, :), W_MRT_un_cols(:, i_tx), W_ZF_un_cols(:, i_tx));
        end
    end

    % Calculate gradient for each lambda_i (user i's lambda)
    for i_user = 1:K_users % Gradient w.r.t. lambda_i (lambda for user i_user)
        % Term 1: Contribution from C_i's own signal term
        % (d/dlambda_i of |h_i * w_i(lambda_i)|^2) / (SINR_i_denominator * ln(2))
        % Numerator of Term 1: dF_dlambda_i(i_user, i_user)
        % Denominator of Term 1: (sigma2_noise + sum of interference to user i_user + signal of user i_user)
        % Interference to user i_user is sum_{j~=i_user} F_ki(i_user, j)
        sum_interference_plus_signal_i = sum(F_ki(i_user, :)); % This is sum_{j=1 to K} |h_i * w_j|^2
        term1_num = dF_dlambda_i(i_user, i_user);
        term1_den = sigma2_noise + sum_interference_plus_signal_i;
        term1 = term1_num / term1_den;

        % Term 2: Contribution from C_k's interference term (k ~= i)
        % Sum over k (k~=i_user) [ (d/dlambda_i of |h_k * w_i(lambda_i)|^2) * |h_k * w_k(lambda_k)|^2 / (complex den) ]
        term2_sum = 0;
        for k_rx_other = 1:K_users
            if k_rx_other == i_user
                continue; % Skip user i_user itself for this term
            end
            % Numerator part for user k_rx_other's rate term affected by lambda_i
            % This is (df_ki / dlambda_i) * f_kk
            % df_ki / dlambda_i is dF_dlambda_i(k_rx_other, i_user)
            % f_kk (signal for user k_rx_other) is F_ki(k_rx_other, k_rx_other)
            term2_num_contrib = dF_dlambda_i(k_rx_other, i_user) * F_ki(k_rx_other, k_rx_other);

            % Denominator part for user k_rx_other's rate term
            % (sigma2 + sum_{j~=k_rx_other} |h_k_rx_other * w_j|^2) * (sigma2 + sum_{j=1 to K} |h_k_rx_other * w_j|^2)
            interference_to_k_rx_other = 0;
            for j_tx_interferer = 1:K_users
                if j_tx_interferer ~= k_rx_other
                    interference_to_k_rx_other = interference_to_k_rx_other + F_ki(k_rx_other, j_tx_interferer);
                end
            end
            den_part1_k_rx_other = sigma2_noise + interference_to_k_rx_other;
            den_part2_k_rx_other = sigma2_noise + sum(F_ki(k_rx_other, :)); % Total received power at k_rx_other + noise

            if abs(den_part1_k_rx_other * den_part2_k_rx_other) > 1e-12 % Avoid division by zero
                term2_sum = term2_sum + (term2_num_contrib / (den_part1_k_rx_other * den_part2_k_rx_other));
            end
        end
        grad_vec(i_user) = (term1 - term2_sum) / log(2); % From user's formula structure
    end
end

function df_val = df_ki_dlambda_analytic(lambda_i_val, hk_vec, w_mrt_i_col, w_zf_i_col)
    % lambda_i_val: scalar lambda for user i
    % hk_vec: 1xNt channel vector for user k
    % w_mrt_i_col, w_zf_i_col: Nt x 1 unnormalized MRT and ZF vectors for user i

    % Unnormalized combined vector for user i
    w_un_i = lambda_i_val * w_mrt_i_col + (1 - lambda_i_val) * w_zf_i_col;
    norm_w_un_i = norm(w_un_i, 2);

    if norm_w_un_i < 1e-10 % If norm is zero, derivative is likely zero or undefined
        df_val = 0;
        return;
    end

    % Derivative of the unnormalized vector w.r.t lambda_i
    dw_un_i_dlambda_i = w_mrt_i_col - w_zf_i_col;

    % Derivative of the norm w.r.t lambda_i
    % d(sqrt(X'X))/dlambda = (1/(2*norm)) * d(X'X)/dlambda = (1/norm) * real(X' * dX/dlambda)
    d_norm_w_un_i_dlambda_i = real(w_un_i' * dw_un_i_dlambda_i) / norm_w_un_i;
    
    % Let f_ki = |h_k * w_i_norm|^2 = |h_k * w_un_i / norm_w_un_i|^2
    % Let num_f = h_k * w_un_i
    % Let den_f = norm_w_un_i
    % f_ki = (num_f * conj(num_f)) / (den_f^2)
    
    num_f = hk_vec * w_un_i;
    den_f = norm_w_un_i;
    
    % Derivative of num_f w.r.t lambda_i
    d_num_f_dlambda_i = hk_vec * dw_un_i_dlambda_i;
    
    % Derivative of den_f w.r.t lambda_i is d_norm_w_un_i_dlambda_i
    
    % Using quotient rule for |A/B|^2 = (A A*)/(B B*)
    % d/dx ( (N N*)/(D D*) ) = [ (dN N* + N dN*) D D* - N N* (dD D* + D dD*) ] / (D D*)^2
    % where dN = d_num_f_dlambda_i, dD = d_norm_w_un_i_dlambda_i
    
    term_num_deriv_part = 2 * real(conj(num_f) * d_num_f_dlambda_i); % Corresponds to d(num_f * conj(num_f))/dlambda_i
    term_den_deriv_part = 2 * den_f * d_norm_w_un_i_dlambda_i;      % Corresponds to d(den_f^2)/dlambda_i
    
    df_val = (term_num_deriv_part * (den_f^2) - (abs(num_f)^2) * term_den_deriv_part) / (den_f^4);

    if isnan(df_val) || isinf(df_val) % Safety check
        df_val = 0;
    end
end

function [C_sum_val, C_vec] = analytic_calc_rate(H_chan, W_pu_norm, sigma2_noise)
    % H_chan: K x Nt channel
    % W_pu_norm: Nt x K precoding matrix, columns are independently normalized
    % sigma2_noise: noise variance
    [K_users, ~] = size(H_chan);
    C_vec = zeros(K_users, 1);
    for k_rx = 1:K_users
        signal_power = abs(H_chan(k_rx, :) * W_pu_norm(:, k_rx))^2;
        interference_power = 0;
        for j_tx_interferer = 1:K_users
            if j_tx_interferer ~= k_rx
                interference_power = interference_power + abs(H_chan(k_rx, :) * W_pu_norm(:, j_tx_interferer))^2;
            end
        end
        sinr_k = signal_power / (interference_power + sigma2_noise);
        if sinr_k < 0 || isnan(sinr_k) || isinf(sinr_k)
            C_vec(k_rx) = 0; % Handle invalid SINR
        else
            C_vec(k_rx) = log2(1 + sinr_k);
        end
    end
    C_sum_val = sum(C_vec);
    if isnan(C_sum_val) || isinf(C_sum_val)
        C_sum_val = 0; % Ensure sum is valid
    end
end
% =========================================================================
%% Interface Function to Apply Precoding (including normalization)
% MODIFIED: Signature changed back, returns optimized lambda
% =========================================================================
function [W, lambda_optimized_vec] = apply_precoding(H, noise_variance, precoder_type, params)
    % Calculates and normalizes the precoding matrix based on the type.
    % For Hybrid_PerUser_Opt, performs multi-dim gradient ascent
    % and returns the optimized lambda vector.
    % Output:
    %   W: Normalized precoding matrix (Nt x K)
    %   lambda_optimized_vec: Vector of optimized lambda values [lambda_1*, ..., lambda_K*]^T
    %                         for Hybrid_PerUser_Opt, NaN vector otherwise.

    Nt = params.Nt;
    K = params.K;
    lambda_optimized_vec = NaN(K, 1); % Default return value

    switch upper(precoder_type)
        case {'ZF', 'RZF', 'MMSE', 'MRT'} % Handle standard cases first
             % --- Calculate W_unnormalized for standard schemes ---
             switch upper(precoder_type)
                case 'ZF'
                    HH_H = H * H';
                    if cond(HH_H) < 1e10
                        W_unnormalized = H' / HH_H;
                    else
                         warning('ZF: Channel matrix H*H^H is ill-conditioned. Using MRT fallback.');
                         W_unnormalized = H';
                    end
                case 'RZF'
                    regularization_rzf = K * noise_variance;
                    W_unnormalized = H' / (H * H' + regularization_rzf * eye(K));
                case 'MMSE'
                    regularization_mmse = K * noise_variance / params.total_power;
                    W_unnormalized = H' / (H * H' + regularization_mmse * eye(K));
                case 'MRT'
                    W_unnormalized = H';
             end
             W = normalize_precoder(W_unnormalized, params.total_power); % Normalize

        case 'HYBRID_ZF_MRT' % <-- NEW CASE
                       % 1. "Train" the lambda vector via optimization and get base precoders
            [lambda_optimized_vec, W_mrt_un_cols, W_zf_un_cols, zf_col_possible] = ...
                optimize_hybrid_per_user_lambda(H, noise_variance, params); % Call the optimization/training function

            % 2. Apply the "trained" lambdas to construct the final precoder
            W = apply_hybrid_per_user_precoding(lambda_optimized_vec, ...
                                                W_mrt_un_cols, W_zf_un_cols, ...
                                                zf_col_possible, params);
        case 'HYBRID_ZF_MRT_02' % <-- NEW CASE: Per-User Normalization Hybrid Scheme
                    % 1. Optimize lambda vector using per-user normalization assumption
                    [lambda_optimized_vec, W_mrt_un_cols, W_zf_un_cols, zf_col_possible] = ...
                        optimize_hybrid_per_user_norm_lambda(H, noise_variance, params); % Uses per-user norm internally
        
                    % 2. Construct final W using per-user normalization
                    W = apply_hybrid_per_user_norm_precoding(lambda_optimized_vec, ...
                                                             W_mrt_un_cols, W_zf_un_cols, ...
                                                             zf_col_possible, params); % Applies per-user norm
        case 'HYBRID_ZF_MRT_03' % <-- NEW CASE for user's method
            % Call the wrapper for the user's analytic gradient based method
            [lambda_optimized_vec, W] = run_hybrid_analytic_indnorm(H, noise_variance, params);

        otherwise
            error('Unknown precoder type: %s', precoder_type);
    end
end


% =========================================================================
%% Function to Calculate Performance Metrics (Sum Rate and BER)
% MODIFIED: Uses custom_qammod and custom_qamdemod
% =========================================================================
% 调制现在在此函数内部完成
function perf_metrics = calculate_ber_metrics(H, W, noise_variance, tx_bits, params)
    % Calculates Sum Rate and Bit Error Rate using custom modulation/demodulation.
    % Inputs:
    %   H: Channel matrix (K x Nt)
    %   W: Normalized precoding matrix (Nt x K)
    %   noise_variance: Noise variance (sigma^2)
    %   tx_symbols_placeholder: NOT USED (was tx_symbols from qammod)
    %   tx_bits: Original bits transmitted (K*bits_per_symbol x num_symbols)
    %   params: Structure with system parameters
    % Output:
    %   perf_metrics: Structure containing .sum_rate, .error_count, .bits_processed

    num_symbols = size(tx_bits, 2); % Number of symbols based on bit matrix columns
    perf_metrics = struct('sum_rate', 0, 'error_count', 0, 'bits_processed', 0);

    % --- Custom Modulation Step ---
    tx_symbols = zeros(params.K, num_symbols); % Initialize symbol matrix
    for k = 1:params.K
        % Extract bits for user k
        user_bits = tx_bits((k-1)*params.bits_per_symbol + 1 : k*params.bits_per_symbol, :);
        % Modulate using custom function
        tx_symbols(k,:) = custom_qammod(user_bits, params.mod_order, true); % UnitAveragePower = true
    end

    % --- 2. BER Calculation (Simulation - Modified Receiver/Demod) ---
    % error_count_calc = 0;
    H_eff = H * W; % Effective channel (K x K)
    bits_processed_calc = params.K * params.bits_per_symbol * num_symbols;

    % Transmit signal through channel (Unchanged)
    tx_signal = W * tx_symbols; % Precoding (Nt x num_symbols)
    noise = sqrt(noise_variance / 2) * (randn(params.K, num_symbols) + 1i * randn(params.K, num_symbols));
    rx_signal = H * tx_signal + noise; % Received signal at users (K x num_symbols)

    % Receiver Processing (Simple Scaling Receiver per User - Unchanged)
    rx_symbols_est = zeros(params.K, num_symbols);
    for k = 1:params.K
        effective_gain = H_eff(k, k);
        if abs(effective_gain) > 1e-9
            rx_symbols_est(k, :) = rx_signal(k, :) / effective_gain;
        else
            rx_symbols_est(k, :) = 0;
        end
    end

    % --- Custom Demodulation and Bit Error Counting ---
    rx_bits = zeros(size(tx_bits));
     for k = 1:params.K
        % Demodulate received symbols for user k using custom function
        rx_bits_k = custom_qamdemod(rx_symbols_est(k,:), params.mod_order, true); % Assumes unit average power constellation
        rx_bits((k-1)*params.bits_per_symbol + 1 : k*params.bits_per_symbol, :) = rx_bits_k;
     end

    % Count errors by comparing tx_bits and rx_bits (Unchanged)
    error_count_calc = sum(tx_bits ~= rx_bits, 'all');

    perf_metrics.error_count = error_count_calc;
    perf_metrics.bits_processed = bits_processed_calc;

end
function perf_metrics = calculate_sumrate_metrics(H, W, noise_variance, params)
    % Calculates Sum Rate and Bit Error Rate using custom modulation/demodulation.
    % Inputs:
    %   H: Channel matrix (K x Nt)
    %   W: Normalized precoding matrix (Nt x K)
    %   noise_variance: Noise variance (sigma^2)
    %   params: Structure with system parameters
    % Output:
    %   perf_metrics: Structure containing .sum_rate, .error_count, .bits_processed

    perf_metrics = struct('sum_rate', 0, 'error_count', 0, 'bits_processed', 0);
    % --- 1. Sum Rate Calculation (Theoretical - Unchanged) ---
    sum_rate_calc = 0;
    H_eff = H * W; % Effective channel (K x K)
    for k = 1:params.K
        signal_power = abs(H_eff(k, k))^2;
        interference_power = norm(H_eff(k, :))^2 - signal_power;
        interference_power = max(0, interference_power); % Ensure non-negative
        sinr_k = signal_power / (interference_power + noise_variance);
        if isnan(sinr_k) || isinf(sinr_k) || sinr_k < 0
             rate_k = 0;
        else
            rate_k = log2(1 + sinr_k);
        end
        sum_rate_calc = sum_rate_calc + rate_k;
    end
    perf_metrics.sum_rate = sum_rate_calc;

end
% =========================================================================
% NEW Helper Function: Custom QAM Modulator (White-Box)
% =========================================================================
function symbols = custom_qammod(bits, M, unit_avg_power)
    % Custom QAM modulator.
    % Inputs:
    %   bits: Bit matrix (log2(M) x N), each column contains bits for one symbol.
    %   M: Modulation order (e.g., 4, 16, 64). Must be a perfect square >= 4.
    %   unit_avg_power: Boolean, if true, normalize constellation to have average power of 1.
    % Output:
    %   symbols: Modulated complex symbols (1 x N).

    bps = log2(M); % Bits per symbol
    if mod(bps, 1) ~= 0 || M < 4 || mod(sqrt(M), 1) ~= 0
        error('Modulation order M must be a square number >= 4 (e.g., 4, 16, 64).');
    end
    if size(bits, 1) ~= bps
        error('Input bit matrix must have log2(M) rows.');
    end

    num_symbols = size(bits, 2);
    symbols = zeros(1, num_symbols);
    sqrtM = sqrt(M);
    % Create mapping levels (e.g., for 16-QAM: [-3, -1, 1, 3])
    levels = -(sqrtM - 1) : 2 : (sqrtM - 1);

    % Generate the reference constellation points (natural binary mapping)
    ref_constellation = zeros(1, M);
    idx = 0;
    for q = 1:sqrtM % Quadrature component index
        for i = 1:sqrtM % In-phase component index
            idx = idx + 1;
            ref_constellation(idx) = levels(i) + 1i * levels(q);
        end
    end

    % Calculate average power if normalization is needed
    if unit_avg_power
        avg_power = mean(abs(ref_constellation).^2);
        scale_factor = 1 / sqrt(avg_power);
    else
        scale_factor = 1;
    end

    % Apply scaling to reference constellation
    ref_constellation = ref_constellation * scale_factor;

    % --- Map bits to symbols ---
    % Convert each column of bits to an integer index (0 to M-1)
    % Assumes natural binary mapping (not Gray coding)
    bit_indices = zeros(1, num_symbols);
    powers_of_2 = 2.^( (bps-1) : -1 : 0 )'; % [2^(bps-1), ..., 2^1, 2^0]'
    for i = 1:num_symbols
       bit_indices(i) = sum(bits(:, i) .* powers_of_2);
    end

    % Map indices to constellation points
    % Indices are 0 to M-1, MATLAB indexing is 1 to M
    symbols = ref_constellation(bit_indices + 1);

end

% =========================================================================
% NEW Helper Function: Custom QAM Demodulator (White-Box)
% =========================================================================
function bits = custom_qamdemod(rx_symbols, M, unit_avg_power)
    % Custom QAM demodulator (Hard Decision, Minimum Euclidean Distance).
    % Inputs:
    %   rx_symbols: Received complex symbols (1 x N).
    %   M: Modulation order (e.g., 4, 16, 64). Must be a perfect square >= 4.
    %   unit_avg_power: Boolean, indicates if the reference constellation used
    %                   at the transmitter had unit average power.
    % Output:
    %   bits: Demodulated bit matrix (log2(M) x N).

    bps = log2(M);
    if mod(bps, 1) ~= 0 || M < 4 || mod(sqrt(M), 1) ~= 0
        error('Modulation order M must be a square number >= 4 (e.g., 4, 16, 64).');
    end

    num_symbols = length(rx_symbols);
    bits = zeros(bps, num_symbols);
    sqrtM = sqrt(M);
    levels = -(sqrtM - 1) : 2 : (sqrtM - 1);

    % Regenerate the reference constellation (same as in modulator)
    ref_constellation = zeros(1, M);
    ref_indices_bits = zeros(bps, M); % Store bits corresponding to each index
    idx = 0;
    powers_of_2 = 2.^( (bps-1) : -1 : 0 )';
    for q = 1:sqrtM
        for i = 1:sqrtM
            idx = idx + 1;
            ref_constellation(idx) = levels(i) + 1i * levels(q);
            % Store corresponding bits (natural binary mapping)
            ref_indices_bits(:, idx) = de2bi(idx-1, bps, 'left-msb')'; % Get bits for index (idx-1)
        end
    end

    % Apply scaling if needed
    if unit_avg_power
        avg_power = mean(abs(ref_constellation).^2);
        scale_factor = 1 / sqrt(avg_power);
    else
        scale_factor = 1;
    end
    ref_constellation = ref_constellation * scale_factor;

    % --- Minimum Euclidean Distance Demodulation ---
    for i = 1:num_symbols
        % Calculate squared Euclidean distances to all reference points
        distances_sq = abs(rx_symbols(i) - ref_constellation).^2;
        % Find the index of the minimum distance
        [~, min_idx] = min(distances_sq);
        % Get the bits corresponding to the closest constellation point
        bits(:, i) = ref_indices_bits(:, min_idx);
    end
end


% =========================================================================
% Function to Plot Results Based on Selected Metric
% MODIFIED: Added 'SpectralEfficiency' case
% =========================================================================
function plot_results(results, params)
    % Plots the simulation results based on params.plot_metric.

    num_schemes = length(params.precoding_schemes);
    markers = {'-o', '-s', '-^', '-d', '-*', '-x'}; % Add more if > 6 schemes
    colors = lines(num_schemes);
    legend_entries = cell(1, num_schemes);
    figure; % Create a new figure for the plot

    plot_type = lower(params.plot_metric);

    % Determine plot function and axis labels based on metric
    switch plot_type
        case {'sumrate', 'spectralefficiency'} % <-- Grouped SumRate and SE
            plot_fcn = @plot;
            y_data_field = 'sum_rate'; % Both use the sum_rate data
            if strcmp(plot_type, 'spectralefficiency')
                 y_label_str = 'Spectral Efficiency (bits/s/Hz)'; % Specific label for SE
                 plot_title = 'Spectral Efficiency Performance';
            else
                 y_label_str = 'Sum Rate / Capacity (bits/s/Hz)'; % Original label
                 plot_title = 'Sum Rate Performance';
            end
            x_label_str = 'SNR (dB)';
            y_scale = 'linear';
            x_scale = 'linear';

        case 'ber'
            plot_fcn = @semilogy;
            y_data_field = 'ber';
            y_label_str = 'Bit Error Rate (BER)';
            x_label_str = 'SNR (dB)';
            y_scale = 'log';
            x_scale = 'linear';
            plot_title = 'BER Performance';
        case 'multiplexinggain'
            plot_fcn = @plot;
            y_data_field = 'sum_rate';
            y_label_str = 'Sum Rate / Capacity (bits/s/Hz)';
            x_label_str = 'SNR (dB)';
            y_scale = 'linear';
            x_scale = 'linear';
            plot_title = 'Sum Rate vs SNR (Observe Multiplexing Gain)';
        case 'diversitygain'
            plot_fcn = @loglog;
            y_data_field = 'ber';
            y_label_str = 'Bit Error Rate (BER)';
            x_label_str = 'SNR (linear scale)';
            y_scale = 'log';
            x_scale = 'log';
            plot_title = 'BER vs Linear SNR (Observe Diversity Gain)';
        otherwise
            error('Unknown plot metric: %s', params.plot_metric);
    end

    hold on;
    for scheme_idx = 1:num_schemes
        scheme_name = params.precoding_schemes{scheme_idx};
        marker_style = markers{mod(scheme_idx-1, length(markers)) + 1};
        y_data = results.(scheme_name).(y_data_field);

        % Prepare x_data based on plot type
        if strcmp(plot_type, 'diversitygain')
             x_data = 10.^(params.SNR_dB_vec / 10); % Linear SNR for loglog diversity plot
        else
             x_data = params.SNR_dB_vec; % dB SNR for other plots
        end

        % Handle cases where BER might be zero for log scale
        if strcmp(y_scale, 'log')
            y_data(y_data <= 0) = NaN;
        end

        % Plot the data
        plot_fcn(x_data, y_data, marker_style, ...
                 'LineWidth', 1.5, 'MarkerSize', 6, 'Color', colors(scheme_idx,:));
        legend_entries{scheme_idx} = sprintf('%s', scheme_name);
    end
    hold off;

    grid on;
    xlabel(x_label_str);
    ylabel(y_label_str);

    % Explicitly set axis scaling (loglog/semilogy handle their own)
    set(gca, 'YScale', y_scale);
    set(gca, 'XScale', x_scale);


    % Set reasonable Y-limits for BER plots
    if strcmp(plot_type, 'ber') || strcmp(plot_type, 'diversitygain')
        min_ber_overall = 1; % Start with max possible BER
        all_ber_positive = [];
         for scheme_idx = 1:num_schemes % Find min non-zero BER across all schemes
              current_ber = results.(params.precoding_schemes{scheme_idx}).ber;
              all_ber_positive = [all_ber_positive, current_ber(current_ber > 0)]; %#ok<AGROW>
         end
         if ~isempty(all_ber_positive)
             min_ber_overall = min(all_ber_positive);
         else
             min_ber_overall = 1e-7; % Default if no errors found at all
         end
         ylim([max(min_ber_overall*0.1, 1e-7) 1]); % Adjust ylim
    end


    % Add title with system parameters
    title_str = sprintf('%s (Nt=%d, K=%d, %s Ch', ...
                        plot_title, params.Nt, params.K, params.channel_model);
    if strcmpi(params.channel_model, 'Rician')
        title_str = [title_str, sprintf(' K=%.1f', params.rician_K_factor)];
    end
     if strcmp(plot_type, 'ber') || strcmp(plot_type, 'diversitygain')
         title_str = [title_str, sprintf(', %d-QAM', params.mod_order)];
     end
    title_str = [title_str, ')'];
    title(title_str);
    legend(legend_entries, 'Location', 'Best');

end

