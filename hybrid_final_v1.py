import numpy as np
import matplotlib.pyplot as plt
import math
import time
import os # 新增: 用于路径操作
import torch # 新增: PyTorch库
import torch.nn as nn # 新增: PyTorch神经网络模块

# =========================================================================
# Python仿真MIMO下行链路预编码性能
# =========================================================================
# 作者: Gemini (由MATLAB代码转换)
# 日期: 2025-05-26
# 版本: 3.7 (每用户混合ZF/MRT预编码优化)
# 描述: 实现一种混合ZF/MRT方案，其中每个用户具有独立的权重lambda_k，
#       通过具有自适应学习率的多维梯度上升进行优化。
# =========================================================================

class SimulationParameters:
    def __init__(self):
        # --- 系统参数 ---
        self.Nt = 155  # 基站发射天线数 (为更快的BER仿真而减少)
        self.K = 150   # 单天线用户数 (为更快的BER仿真而减少)
        self.total_power = 1.0  # 基站总发射功率约束

        # --- 信道模型参数 ---
        self.channel_model = 'Rayleigh'  # 'Rayleigh' 或 'Rician'
        self.rician_K_factor = 10.0    # 莱斯K因子 (线性尺度), 仅用于 'Rician'

        # --- 调制和BER参数 ---
        self.mod_order = 4  # 调制阶数 (例如, 4 表示 QPSK, 16 表示 16-QAM)
        self.bits_per_symbol = int(np.log2(self.mod_order))
        self.num_bits_per_symbol_target = 1e4 # 每个SNR点每个用户的目标比特数
                                              # 增加可获得更平滑的BER曲线，减少可加快速度。
        self.num_symbols_per_run = math.ceil(self.num_bits_per_symbol_target / self.bits_per_symbol)

        # --- 仿真控制 ---
        self.SNR_dB_vec = np.arange(-20, 21, 1)  # SNR范围 (dB)
        self.num_monte_carlo = 10  # 蒙特卡洛信道实现次数。初始为10
                                   # BER平均在MC循环内部对符号进行。

        # --- 预编码方案选择 ---
        self.precoding_schemes = ['ZF', 'MMSE', 'MRT', 'Hybrid_ZF_MRT_03', 'LCGNET_CG',  'Hybrid_ZF_MRT_LCGNET_CG'] # 新增Hybrid_ZF_MRT_LCGNET_CG方案
        # 'ZF', 'MMSE', 'MRT', 'Hybrid_ZF_MRT_03', 'LCGNET_CG', 'LCGNET_Richardson', 'Hybrid_ZF_MRT_LCGNET_CG'

        # --- 混合预编码优化参数 ---
        self.hybrid = {
            'lambda_init': 0.5,         # 初始加权因子 lambda
            'learning_rate_init': 0.1,  # 梯度上升的初始学习率
            'lr_reduction_factor': 10.0,# 当梯度反转时降低学习率的因子
            'max_iterations': 500,     # 每个信道lambda优化的最大迭代次数
            'gradient_delta': 1e-4,     # 用于数值梯度计算的小delta
            'convergence_threshold': 1e-6,# lambda变化声明收敛的阈值
            'grad_clip': 1.0            # 梯度截断阈值
        }
                # --- LCGNET 相关参数 ---
        self.lcgnet_model_dir = "models"
        self.lcgnet_num_layers_richardson = 10
        self.lcgnet_num_layers_cg = 5
        
        self.lcgnet_model_path_cg = os.path.join(self.lcgnet_model_dir, f"CG_dun_mse_k{self.K}_nt{self.Nt}_l{self.lcgnet_num_layers_cg}.pth")
        #self.lcgnet_model_path_richardson = os.path.join(self.lcgnet_model_dir, f"Richardson_dun_mse_k{self.K}_nt{self.Nt}_l{self.lcgnet_num_layers_richardson}.pth")

        self.lcgnet_dtype = torch.complex64
        self.lcgnet_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # --- 可视化控制 ---
        # 选择要绘制的指标:
        self.plot_metric = 'SumRate'  # 'SumRate', 'BER', 'MultiplexingGain', 'DiversityGain', 'SpectralEfficiency'

        self.plot_sumrate = 'true' # 'true' 或 'false'
        self.plot_ber = 'false'    # 'true' 或 'false'
        self.plot_se = 'false'     # 'true' 或 'false'

def generate_channel_model(params: SimulationParameters) -> np.ndarray:
    """
    根据指定的模型生成 K x Nt MIMO信道矩阵。
    """
    if params.channel_model.lower() == 'rayleigh':
        H = (np.random.randn(params.K, params.Nt) + 1j * np.random.randn(params.K, params.Nt)) / np.sqrt(2)
    elif params.channel_model.lower() == 'rician':
        K_factor = params.rician_K_factor
        H_nlos = (np.random.randn(params.K, params.Nt) + 1j * np.random.randn(params.K, params.Nt)) / np.sqrt(2)
        # 简单的固定LoS分量 (每次实现随机)
        H_los = np.exp(1j * 2 * np.pi * np.random.rand(params.K, params.Nt)) # 随机相位LoS
        H = np.sqrt(K_factor / (K_factor + 1)) * H_los + np.sqrt(1 / (K_factor + 1)) * H_nlos
    else:
        raise ValueError(f'未知的信道模型: {params.channel_model}')
    return H

def normalize_precoder(W_un: np.ndarray, P_total: float) -> np.ndarray:
    """
    归一化预编码矩阵W_un以满足 trace(W @ W.conj().T) = P_total。
    """
    norm_W_sq = np.trace(W_un @ W_un.conj().T)
    if np.abs(norm_W_sq) > 1e-10: # 检查范数是否合理非零
        # power_scaling_factor = np.sqrt(P_total / norm_W_sq) # 这应该是复数安全的
        power_scaling_factor = np.sqrt(P_total / np.real(norm_W_sq)) # 功率是实数
        W_norm = power_scaling_factor * W_un
    else:
        # 如果范数为零或非常小，返回零矩阵
        Nt, K_users = W_un.shape
        W_norm = np.zeros((Nt, K_users), dtype=complex)

    # 安全性最终检查
    if np.any(np.isnan(W_norm)) or np.any(np.isinf(W_norm)):
        Nt, K_users = W_un.shape
        W_norm = np.zeros((Nt, K_users), dtype=complex)
    return W_norm

# --- 混合方案的辅助函数 ---
def analytic_param_precoding(lambda_vec: np.ndarray, W_MRT_un: np.ndarray, W_ZF_un: np.ndarray) -> np.ndarray:
    """
    lambda_vec: Kx1 的 lambda 向量
    W_MRT_un: Nt x K 未归一化的MRT矩阵
    W_ZF_un: Nt x K 未归一化的ZF矩阵
    输出: W_norm_pu: Nt x K 矩阵, 每列独立归一化为1
    """
    Nt, K = W_MRT_un.shape
    W_norm_pu = np.zeros((Nt, K), dtype=complex)
    for i in range(K):
        w_un_i = lambda_vec[i] * W_MRT_un[:, i] + (1 - lambda_vec[i]) * W_ZF_un[:, i]
        norm_wi = np.linalg.norm(w_un_i, 2)
        if norm_wi > 1e-10:
            W_norm_pu[:, i] = w_un_i / norm_wi
        else:
            W_norm_pu[:, i] = np.zeros(Nt, dtype=complex) # 或其他零范数处理
    return W_norm_pu

def df_ki_dlambda_analytic(lambda_i_val: float, hk_vec: np.ndarray, w_mrt_i_col: np.ndarray, w_zf_i_col: np.ndarray) -> float:
    """
    lambda_i_val: 用户i的标量lambda
    hk_vec: 用户k的1xNt信道向量
    w_mrt_i_col, w_zf_i_col: 用户i的Nt x 1未归一化MRT和ZF向量
    """
    # 用户i的未归一化组合向量
    w_un_i = lambda_i_val * w_mrt_i_col + (1 - lambda_i_val) * w_zf_i_col
    norm_w_un_i = np.linalg.norm(w_un_i, 2)

    if norm_w_un_i < 1e-10: # 如果范数为零，导数可能为零或未定义
        return 0.0

    # 未归一化向量对lambda_i的导数
    dw_un_i_dlambda_i = w_mrt_i_col - w_zf_i_col

    # 范数对lambda_i的导数
    # d(sqrt(X'X))/dlambda = (1/(2*norm)) * d(X'X)/dlambda = (1/norm) * real(X.conj().T @ dX/dlambda)
    d_norm_w_un_i_dlambda_i = np.real(w_un_i.conj().T @ dw_un_i_dlambda_i) / norm_w_un_i
    
    # 令 f_ki = |h_k @ w_i_norm|^2 = |h_k @ w_un_i / norm_w_un_i|^2
    # 令 num_f = h_k @ w_un_i
    # 令 den_f = norm_w_un_i
    # f_ki = (num_f * conj(num_f)) / (den_f^2)
    
    num_f = hk_vec @ w_un_i
    den_f = norm_w_un_i
    
    # num_f 对 lambda_i 的导数
    d_num_f_dlambda_i = hk_vec @ dw_un_i_dlambda_i
    
    # den_f 对 lambda_i 的导数是 d_norm_w_un_i_dlambda_i
    
    # 使用 |A/B|^2 = (A A*)/(B B*) 的商法则
    # d/dx ( (N N*)/(D D*) ) = [ (dN N* + N dN*) D D* - N N* (dD D* + D dD*) ] / (D D*)^2
    # 其中 dN = d_num_f_dlambda_i, dD = d_norm_w_un_i_dlambda_i
    
    term_num_deriv_part = 2 * np.real(np.conj(num_f) * d_num_f_dlambda_i) # 对应于 d(num_f * conj(num_f))/dlambda_i
    term_den_deriv_part = 2 * den_f * d_norm_w_un_i_dlambda_i      # 对应于 d(den_f^2)/dlambda_i
    
    if den_f**4 < 1e-20: # 避免除以非常小的数
        df_val = 0.0
    else:
        df_val = (term_num_deriv_part * (den_f**2) - (np.abs(num_f)**2) * term_den_deriv_part) / (den_f**4)

    if np.isnan(df_val) or np.isinf(df_val): # 安全检查
        df_val = 0.0
    return df_val

def calc_gradient_analytic(H_chan: np.ndarray, W_curr_pu_norm: np.ndarray, W_MRT_un_cols: np.ndarray, W_ZF_un_cols: np.ndarray, lambda_vec: np.ndarray, sigma2_noise: float, params: SimulationParameters) -> np.ndarray:
    """
    H_chan: K x Nt 信道矩阵
    W_curr_pu_norm: Nt x K 当前预编码矩阵 (列独立归一化)
    W_MRT_un_cols, W_ZF_un_cols: Nt x K 未归一化的基础预编码器
    lambda_vec: Kx1 当前lambdas
    sigma2_noise: 噪声方差
    """
    K_users = params.K
    grad_vec = np.zeros(K_users)

    # 预计算所有 f_ki = |h_k @ w_i(lambda_i)|^2 其中 w_i 来自 W_curr_pu_norm
    F_ki = np.zeros((K_users, K_users))
    for k_rx in range(K_users):
        for i_tx in range(K_users):
            F_ki[k_rx, i_tx] = np.abs(H_chan[k_rx, :] @ W_curr_pu_norm[:, i_tx])**2

    # 预计算所有 df_ki/dlambda_i ( |h_k @ w_i(lambda_i)|^2 对 lambda_i 的导数)
    # 其中 w_i(lambda_i) 是每用户归一化向量
    dF_dlambda_i = np.zeros((K_users, K_users)) # dF_dlambda_i[k,i] 表示 df_ki / dlambda_i
    for k_rx in range(K_users):
        for i_tx in range(K_users): # 导数是关于 lambda_i (即 lambda_vec[i_tx])
            dF_dlambda_i[k_rx, i_tx] = df_ki_dlambda_analytic(lambda_vec[i_tx], H_chan[k_rx, :], W_MRT_un_cols[:, i_tx], W_ZF_un_cols[:, i_tx])

    # 计算每个 lambda_i (用户 i 的 lambda) 的梯度
    for i_user in range(K_users): # 关于 lambda_i (用户 i_user 的 lambda) 的梯度
        # 项1: 来自 C_i 自身信号项的贡献
        # (d/dlambda_i of |h_i @ w_i(lambda_i)|^2) / (SINR_i_denominator * ln(2))
        # 项1的分子: dF_dlambda_i[i_user, i_user]
        # 项1的分母: (sigma2_noise + 用户 i_user 的干扰总和 + 用户 i_user 的信号)
        # 用户 i_user 的干扰是 sum_{j~=i_user} F_ki[i_user, j]
        sum_interference_plus_signal_i = np.sum(F_ki[i_user, :]) # 这是 sum_{j=1 to K} |h_i @ w_j|^2
        term1_num = dF_dlambda_i[i_user, i_user]
        term1_den = sigma2_noise + sum_interference_plus_signal_i
        if np.abs(term1_den) < 1e-12: term1 = 0.0
        else: term1 = term1_num / term1_den

        # 项2: 来自 C_k 的干扰项的贡献 (k ~= i)
        # Sum over k (k~=i_user) [ (d/dlambda_i of |h_k @ w_i(lambda_i)|^2) * |h_k @ w_k(lambda_k)|^2 / (复杂分母) ]
        term2_sum = 0.0
        for k_rx_other in range(K_users):
            if k_rx_other == i_user:
                continue # 此项跳过用户 i_user 本身

            # 用户 k_rx_other 的速率项受 lambda_i 影响的分子部分
            # 这是 (df_ki / dlambda_i) * f_kk
            # df_ki / dlambda_i 是 dF_dlambda_i[k_rx_other, i_user]
            # f_kk (用户 k_rx_other 的信号) 是 F_ki[k_rx_other, k_rx_other]
            term2_num_contrib = dF_dlambda_i[k_rx_other, i_user] * F_ki[k_rx_other, k_rx_other]

            # 用户 k_rx_other 的速率项的分母部分
            # (sigma2 + sum_{j~=k_rx_other} |h_k_rx_other @ w_j|^2) * (sigma2 + sum_{j=1 to K} |h_k_rx_other @ w_j|^2)
            interference_to_k_rx_other = 0.0
            for j_tx_interferer in range(K_users):
                if j_tx_interferer != k_rx_other:
                    interference_to_k_rx_other += F_ki[k_rx_other, j_tx_interferer]
            
            den_part1_k_rx_other = sigma2_noise + interference_to_k_rx_other
            den_part2_k_rx_other = sigma2_noise + np.sum(F_ki[k_rx_other, :]) # k_rx_other 处接收的总功率 + 噪声

            if np.abs(den_part1_k_rx_other * den_part2_k_rx_other) > 1e-12: # 避免除以零
                term2_sum += (term2_num_contrib / (den_part1_k_rx_other * den_part2_k_rx_other))
        
        grad_vec[i_user] = (term1 - term2_sum) / np.log(2) # 根据用户公式结构
    return grad_vec

def analytic_calc_rate(H_chan: np.ndarray, W_pu_norm: np.ndarray, sigma2_noise: float):
    """
    H_chan: K x Nt 信道
    W_pu_norm: Nt x K 预编码矩阵, 列独立归一化
    sigma2_noise: 噪声方差
    """
    K_users, _ = H_chan.shape
    C_vec = np.zeros(K_users)
    for k_rx in range(K_users):
        signal_power = np.abs(H_chan[k_rx, :] @ W_pu_norm[:, k_rx])**2
        interference_power = 0.0
        for j_tx_interferer in range(K_users):
            if j_tx_interferer != k_rx:
                interference_power += np.abs(H_chan[k_rx, :] @ W_pu_norm[:, j_tx_interferer])**2
        
        sinr_k = signal_power / (interference_power + sigma2_noise)
        if sinr_k < 0 or np.isnan(sinr_k) or np.isinf(sinr_k):
            C_vec[k_rx] = 0.0 # 处理无效SINR
        else:
            C_vec[k_rx] = np.log2(1 + sinr_k)
            
    C_sum_val = np.sum(C_vec)
    if np.isnan(C_sum_val) or np.isinf(C_sum_val):
        C_sum_val = 0.0 # 确保和有效
    return C_sum_val, C_vec

# --- 修改: run_hybrid_analytic_indnorm 函数，使其接受外部提供的 W_MRT_un 和 W_ZF_un ---
def run_hybrid_analytic_indnorm(H: np.ndarray, sigma2: float, params: SimulationParameters, 
                                W_MRT_un_provided: np.ndarray, W_ZF_un_provided: np.ndarray) :
    """
    用户提供的混合方案的包装器。使用外部提供的W_MRT_un和W_ZF_un。
    H: 信道矩阵
    sigma2: 噪声方差
    params: 主要仿真参数结构
    W_MRT_un_provided: 外部提供的未归一化MRT矩阵
    W_ZF_un_provided: 外部提供的未归一化ZF（或其近似）矩阵
    """
    K = params.K
    # Nt = params.Nt # Nt可以从W_MRT_un_provided的形状中获取

    # 初始化参数，如用户的precoding_module
    eta_vec = np.ones(K) * params.hybrid['learning_rate_init'] # 初始学习率
    lambda_val = params.hybrid['lambda_init'] * np.ones(K)    # 初始lambda
    prev_grad = np.zeros(K)
    prev_rate = -np.inf # 初始化为一个非常小的数

    # 使用外部提供的未归一化预编码矩阵
    W_MRT_un = W_MRT_un_provided
    W_ZF_un = W_ZF_un_provided

    # 迭代优化循环
    for iter_count in range(params.hybrid['max_iterations']):
        W_current = analytic_param_precoding(lambda_val, W_MRT_un, W_ZF_un)
        grad = calc_gradient_analytic(H, W_current, W_MRT_un, W_ZF_un, lambda_val, sigma2, params)
        grad = np.clip(grad, -params.hybrid['grad_clip'], params.hybrid['grad_clip'])
        C_sum_current, _ = analytic_calc_rate(H, W_current, sigma2)
        #动态学习率
        if iter_count > 0:
            sign_changes = (grad * prev_grad) < -1e-12
            rate_decrease = C_sum_current < prev_rate
            for i in range(K):
                if sign_changes[i] or rate_decrease:
                    eta_vec[i] = eta_vec[i] / params.hybrid['lr_reduction_factor']
        prev_grad = grad.copy()
        prev_rate = C_sum_current
        #lambda更新
        lambda_new = lambda_val + eta_vec * grad
        lambda_new = np.clip(lambda_new, 0, 1)

        if np.linalg.norm(lambda_new - lambda_val) < params.hybrid['convergence_threshold']:
            break
        lambda_val = lambda_new.copy()
    
    lambda_opt_vec = lambda_val
    W_opt = analytic_param_precoding(lambda_opt_vec, W_MRT_un, W_ZF_un)
    return lambda_opt_vec, W_opt


def apply_precoding(H: np.ndarray, noise_variance: float, precoder_type: str, params: SimulationParameters, loaded_models: dict) :
    Nt = params.Nt
    K = params.K
    lambda_optimized_vec = np.full(K, np.nan)
    W = np.zeros((Nt, K), dtype=complex)

    if precoder_type.upper() in ['ZF', 'RZF', 'MMSE', 'MRT']:
        # ... (原有ZF, RZF, MMSE, MRT逻辑保持不变，此处省略) ...
        W_unnormalized = np.zeros((Nt, K), dtype=complex)
        if precoder_type.upper() == 'ZF':
            HH_H = H @ H.conj().T
            if np.linalg.cond(HH_H) < 1e10:
                try: W_unnormalized = H.conj().T @ np.linalg.inv(HH_H)
                except np.linalg.LinAlgError: W_unnormalized = H.conj().T @ np.linalg.pinv(HH_H)
            else: W_unnormalized = H.conj().T
        elif precoder_type.upper() == 'RZF': 
            regularization_rzf = K * noise_variance
            W_unnormalized = H.conj().T @ np.linalg.inv(H @ H.conj().T + regularization_rzf * np.eye(K, dtype=complex))
        elif precoder_type.upper() == 'MMSE':
            regularization_mmse = K * noise_variance / params.total_power
            W_unnormalized = H.conj().T @ np.linalg.inv(H @ H.conj().T + regularization_mmse * np.eye(K, dtype=complex))
        elif precoder_type.upper() == 'MRT':
            W_unnormalized = H.conj().T
        W = normalize_precoder(W_unnormalized, params.total_power)

    elif precoder_type.upper() == 'HYBRID_ZF_MRT_03':
        # 计算经典的 W_MRT_un 和 W_ZF_un
        W_MRT_un_classic = H.conj().T
        HH_H_classic = H @ H.conj().T
        try:
            if np.linalg.cond(HH_H_classic) < 1e10:
                W_ZF_un_classic = H.conj().T @ np.linalg.inv(HH_H_classic)
            else:
                # print(f"警告 (Hybrid_ZF_MRT_03): H@H.conj().T 病态，使用伪逆。") # 可以取消注释以进行调试
                W_ZF_un_classic = H.conj().T @ np.linalg.pinv(HH_H_classic)
        except np.linalg.LinAlgError:
            # print(f"警告 (Hybrid_ZF_MRT_03): H@H.conj().T 奇异，使用伪逆。") # 可以取消注释以进行调试
            W_ZF_un_classic = H.conj().T @ np.linalg.pinv(HH_H_classic)
        
        lambda_optimized_vec, W_intermediate = run_hybrid_analytic_indnorm(H, noise_variance, params, W_MRT_un_classic, W_ZF_un_classic)
        W = normalize_precoder(W_intermediate, params.total_power)

    elif precoder_type.upper() in ['LCGNET_CG', 'LCGNET_RICHARDSON']:
        # ... (原有LCGNET_CG, LCGNET_RICHARDSON逻辑保持不变，此处省略) ...
        model_key = precoder_type.upper() 
        lcgnet_model = loaded_models.get(model_key)
        if lcgnet_model is None: # 模型未加载时的回退逻辑
            print(f"警告: {model_key} 模型未加载。将使用ZF作为回退。")
            HH_H = H @ H.conj().T
            if np.linalg.cond(HH_H) < 1e10:
                 try: W_unnormalized = H.conj().T @ np.linalg.inv(HH_H)
                 except np.linalg.LinAlgError: W_unnormalized = H.conj().T @ np.linalg.pinv(HH_H)
            else: W_unnormalized = H.conj().T
            W = normalize_precoder(W_unnormalized, params.total_power)
        else:
            A_matrix = H @ H.conj().T 
            A_torch = torch.tensor(A_matrix, dtype=params.lcgnet_dtype, device=params.lcgnet_device).unsqueeze(0)
            lcgnet_model.eval() 
            with torch.no_grad(): A_inv_approx_torch = lcgnet_model(A_torch)
            A_inv_approx = A_inv_approx_torch.squeeze(0).cpu().numpy()
            W_unnormalized = H.conj().T @ A_inv_approx
            W = normalize_precoder(W_unnormalized, params.total_power)
    
    # --- 新增: Hybrid_ZF_MRT_LCGNET_CG 预编码方案 ---
    elif precoder_type.upper() == 'HYBRID_ZF_MRT_LCGNET_CG':
        lcgnet_cg_model = loaded_models.get('LCGNET_CG')
        if lcgnet_cg_model is None:
            print(f"警告: HYBRID_ZF_MRT_LCGNET_CG 需要 LCGNET_CG 模型，但该模型未加载。将使用 Hybrid_ZF_MRT_03 作为回退。")
            # 回退到 Hybrid_ZF_MRT_03
            W_MRT_un_classic = H.conj().T
            HH_H_classic = H @ H.conj().T
            try:
                if np.linalg.cond(HH_H_classic) < 1e10: W_ZF_un_classic = H.conj().T @ np.linalg.inv(HH_H_classic)
                else: W_ZF_un_classic = H.conj().T @ np.linalg.pinv(HH_H_classic)
            except np.linalg.LinAlgError: W_ZF_un_classic = H.conj().T @ np.linalg.pinv(HH_H_classic)
            lambda_optimized_vec, W_intermediate = run_hybrid_analytic_indnorm(H, noise_variance, params, W_MRT_un_classic, W_ZF_un_classic)
            W = normalize_precoder(W_intermediate, params.total_power)
        else:
            # 1. 计算 W_MRT_un
            W_MRT_un = H.conj().T
            
            # 2. 使用 LCGNET_CG 计算 W_ZF_un 的近似
            A_matrix = H @ H.conj().T
            A_torch = torch.tensor(A_matrix, dtype=params.lcgnet_dtype, device=params.lcgnet_device).unsqueeze(0)
            lcgnet_cg_model.eval()
            with torch.no_grad():
                A_inv_approx_torch = lcgnet_cg_model(A_torch)
            A_inv_approx = A_inv_approx_torch.squeeze(0).cpu().numpy()
            W_ZF_un_lcgnet = H.conj().T @ A_inv_approx
            
            # 3. 调用混合优化函数
            lambda_optimized_vec, W_intermediate = run_hybrid_analytic_indnorm(H, noise_variance, params, W_MRT_un, W_ZF_un_lcgnet)
            W = normalize_precoder(W_intermediate, params.total_power)
    # --- Hybrid_ZF_MRT_LCGNET_CG 预编码方案结束 ---

    else:
        raise ValueError(f'未知的预编码器类型: {precoder_type}')
    
    return W, lambda_optimized_vec


# --- LCGNET 模型定义 (从 LCGNET_precoding_now.py 提取并适配) ---
class RichardsonDUN(nn.Module):
    def __init__(self, num_layers, k_dim, dtype=torch.complex64):
        super(RichardsonDUN, self).__init__()
        self.num_layers = num_layers
        self.k_dim = k_dim
        self.dtype = dtype
        self.omegas = nn.ParameterList([nn.Parameter(torch.rand(1, dtype=torch.float32) * 0.1)
                                          for _ in range(num_layers)])

    def forward(self, A_batch):
        batch_size = A_batch.shape[0]
        X_approx = torch.zeros(batch_size, self.k_dim, self.k_dim, dtype=self.dtype, device=A_batch.device)
        I_k = torch.eye(self.k_dim, dtype=self.dtype, device=A_batch.device).unsqueeze(0).repeat(batch_size, 1, 1)
        for i in range(self.num_layers):
            omega = self.omegas[i].to(self.dtype)
            residual_term = I_k - torch.matmul(A_batch, X_approx)
            X_approx = X_approx + omega * residual_term
        return X_approx

class CGDUN(nn.Module):
    def __init__(self, num_layers, k_dim, dtype=torch.complex64):
        super(CGDUN, self).__init__()
        self.num_layers = num_layers
        self.k_dim = k_dim
        self.dtype = dtype
        self.alphas = nn.ParameterList([nn.Parameter(torch.rand(1, dtype=torch.float32) * 0.1)
                                          for _ in range(num_layers)])
        self.betas = nn.ParameterList([nn.Parameter(torch.rand(1, dtype=torch.float32) * 0.01)
                                         for _ in range(num_layers)])

    def forward(self, A_batch):
        batch_size = A_batch.shape[0]
        X_approx = torch.zeros(batch_size, self.k_dim, self.k_dim, dtype=self.dtype, device=A_batch.device)
        I_k = torch.eye(self.k_dim, dtype=self.dtype, device=A_batch.device).unsqueeze(0).repeat(batch_size, 1, 1)
        r_current = I_k - torch.matmul(A_batch, X_approx)
        p_current = r_current.clone()
        for i in range(self.num_layers):
            alpha_param = self.alphas[i].to(self.dtype)
            beta_param = self.betas[i].to(self.dtype)
            Ap = torch.matmul(A_batch, p_current)
            X_approx = X_approx + alpha_param * p_current
            r_next = r_current - alpha_param * Ap
            p_current = r_next + beta_param * p_current
            r_current = r_next
        return X_approx
# --- LCGNET 模型定义结束 ---


# --- QAM调制/解调函数 ---
def custom_qammod(bits: np.ndarray, M: int, unit_avg_power: bool) -> np.ndarray:
    """
    自定义QAM调制器。
    输入:
      bits: 比特矩阵 (log2(M) x N), 每列包含一个符号的比特。
      M: 调制阶数 (例如, 4, 16, 64)。必须是 >= 4 的完全平方数。
      unit_avg_power:布尔值，如果为true，则将星座归一化为平均功率为1。
    输出:
      symbols: 调制的复数符号 (1 x N)。
    """
    bps = int(np.log2(M))
    if not (M >= 4 and bps == np.log2(M) and np.sqrt(M) == int(np.sqrt(M))): # 检查M是否为4, 16, 64等
        raise ValueError('调制阶数M必须是 >= 4 的平方数 (例如, 4, 16, 64)。')
    if bits.shape[0] != bps:
        raise ValueError(f'输入比特矩阵必须有 {bps} 行。')

    num_symbols = bits.shape[1]
    symbols = np.zeros(num_symbols, dtype=complex)
    sqrtM = int(np.sqrt(M))
    levels = np.arange(-(sqrtM - 1), sqrtM, 2) # 例如, 对于16-QAM: [-3, -1, 1, 3]

    # 生成参考星座点 (自然二进制映射)
    ref_constellation = np.zeros(M, dtype=complex)
    idx = 0
    for q_idx in range(sqrtM): # 正交分量索引
        for i_idx in range(sqrtM): #同相分量索引
            ref_constellation[idx] = levels[i_idx] + 1j * levels[q_idx]
            idx += 1
    
    if unit_avg_power:
        avg_power = np.mean(np.abs(ref_constellation)**2)
        scale_factor = 1 / np.sqrt(avg_power) if avg_power > 1e-9 else 1.0
    else:
        scale_factor = 1.0
    
    ref_constellation *= scale_factor

    # --- 将比特映射到符号 ---
    # 将每列比特转换为整数索引 (0 到 M-1)
    # 假设自然二进制映射 (非格雷编码)
    powers_of_2 = 2**np.arange(bps - 1, -1, -1) # [2^(bps-1), ..., 2^1, 2^0]
    
    for i in range(num_symbols):
       bit_indices_val = np.sum(bits[:, i] * powers_of_2)
       symbols[i] = ref_constellation[bit_indices_val]
       
    return symbols


def custom_qamdemod(rx_symbols: np.ndarray, M: int, unit_avg_power: bool) -> np.ndarray:
    """
    自定义QAM解调器 (硬判决, 最小欧氏距离)。
    输入:
      rx_symbols: 接收到的复数符号 (1 x N)。
      M: 调制阶数 (例如, 4, 16, 64)。必须是 >= 4 的完全平方数。
      unit_avg_power: 布尔值，指示发送端使用的参考星座是否具有单位平均功率。
    输出:
      bits: 解调后的比特矩阵 (log2(M) x N)。
    """
    bps = int(np.log2(M))
    if not (M >= 4 and bps == np.log2(M) and np.sqrt(M) == int(np.sqrt(M))):
        raise ValueError('调制阶数M必须是 >= 4 的平方数 (例如, 4, 16, 64)。')

    num_symbols = len(rx_symbols)
    demod_bits = np.zeros((bps, num_symbols), dtype=int)
    sqrtM = int(np.sqrt(M))
    levels = np.arange(-(sqrtM - 1), sqrtM, 2)

    # 重新生成参考星座 (与调制器中相同)
    ref_constellation = np.zeros(M, dtype=complex)
    ref_indices_bits = np.zeros((bps, M), dtype=int) # 存储对应于每个索引的比特
    
    idx = 0
    powers_of_2 = 2**np.arange(bps - 1, -1, -1) # For de2bi equivalent
    
    for q_idx in range(sqrtM):
        for i_idx in range(sqrtM):
            ref_constellation[idx] = levels[i_idx] + 1j * levels[q_idx]
            # 存储相应的比特 (自然二进制映射)
            # de2bi(idx, bps, 'left-msb')' 的 Python 等效
            binary_representation = bin(idx)[2:].zfill(bps) # '0b' 前缀, 用0填充
            ref_indices_bits[:, idx] = [int(b) for b in binary_representation]
            idx += 1
            
    if unit_avg_power:
        avg_power = np.mean(np.abs(ref_constellation)**2)
        scale_factor = 1 / np.sqrt(avg_power) if avg_power > 1e-9 else 1.0
    else:
        scale_factor = 1.0
    ref_constellation *= scale_factor

    # --- 最小欧氏距离解调 ---
    for i in range(num_symbols):
        # 计算到所有参考点的平方欧氏距离
        distances_sq = np.abs(rx_symbols[i] - ref_constellation)**2
        # 找到最小距离的索引
        min_idx = np.argmin(distances_sq)
        # 获取与最近星座点对应的比特
        demod_bits[:, i] = ref_indices_bits[:, min_idx]
        
    return demod_bits

# --- 性能度量计算函数 ---
def calculate_ber_metrics(H: np.ndarray, W: np.ndarray, noise_variance: float, tx_bits: np.ndarray, params: SimulationParameters) -> dict:
    """
    使用自定义调制/解调计算误比特率。
    输入:
      H: 信道矩阵 (K x Nt)
      W: 归一化的预编码矩阵 (Nt x K)
      noise_variance: 噪声方差 (sigma^2)
      tx_bits: 原始发送比特 (K*bits_per_symbol x num_symbols)
      params: 包含系统参数的结构
    输出:
      perf_metrics: 包含 .error_count, .bits_processed 的字典
    """
    num_symbols = tx_bits.shape[1] # 基于比特矩阵列数的符号数
    perf_metrics = {'error_count': 0, 'bits_processed': 0}

    # --- 自定义调制步骤 ---
    tx_symbols = np.zeros((params.K, num_symbols), dtype=complex) # 初始化符号矩阵
    for k in range(params.K):
        # 提取用户k的比特
        user_bits = tx_bits[k*params.bits_per_symbol : (k+1)*params.bits_per_symbol, :]
        # 使用自定义函数进行调制
        tx_symbols[k,:] = custom_qammod(user_bits, params.mod_order, True) # UnitAveragePower = true

    # --- BER计算 (仿真 - 修改后的接收机/解调) ---
    H_eff = H @ W # 有效信道 (K x K)
    bits_processed_calc = params.K * params.bits_per_symbol * num_symbols

    # 通过信道传输信号
    tx_signal = W @ tx_symbols # 预编码 (Nt x num_symbols)
    noise = np.sqrt(noise_variance / 2) * (np.random.randn(params.K, num_symbols) + 1j * np.random.randn(params.K, num_symbols))
    rx_signal = H @ tx_signal + noise # 用户处接收信号 (K x num_symbols)

    # 接收机处理 (每用户简单缩放接收机)
    rx_symbols_est = np.zeros((params.K, num_symbols), dtype=complex)
    for k in range(params.K):
        effective_gain = H_eff[k, k]
        if np.abs(effective_gain) > 1e-9:
            rx_symbols_est[k, :] = rx_signal[k, :] / effective_gain
        else:
            rx_symbols_est[k, :] = 0 # 或者用噪声填充

    # --- 自定义解调和误比特计数 ---
    rx_bits = np.zeros_like(tx_bits, dtype=int)
    for k in range(params.K):
        # 使用自定义函数解调用户k的接收符号
        rx_bits_k = custom_qamdemod(rx_symbols_est[k,:], params.mod_order, True) # 假设单位平均功率星座
        rx_bits[k*params.bits_per_symbol : (k+1)*params.bits_per_symbol, :] = rx_bits_k

    # 通过比较tx_bits和rx_bits来计数错误
    error_count_calc = np.sum(tx_bits != rx_bits)

    perf_metrics['error_count'] = error_count_calc
    perf_metrics['bits_processed'] = bits_processed_calc
    return perf_metrics

def calculate_sumrate_metrics(H: np.ndarray, W: np.ndarray, noise_variance: float, params: SimulationParameters) -> dict:
    """
    计算和速率。
    输入:
      H: 信道矩阵 (K x Nt)
      W: 归一化的预编码矩阵 (Nt x K)
      noise_variance: 噪声方差 (sigma^2)
      params: 包含系统参数的结构
    输出:
      perf_metrics: 包含 .sum_rate 的字典
    """
    perf_metrics = {'sum_rate': 0.0}
    sum_rate_calc = 0.0
    H_eff = H @ W # 有效信道 (K x K)
    for k in range(params.K):
        signal_power = np.abs(H_eff[k, k])**2
        # interference_power = np.linalg.norm(H_eff[k, :])**2 - signal_power # 这可能不准确，因为H_eff[k,:]包含了信号项
        interference_terms = H_eff[k, np.arange(params.K) != k] # 所有非k列
        interference_power = np.sum(np.abs(interference_terms)**2)

        interference_power = max(0, interference_power) # 确保非负
        
        if (interference_power + noise_variance) < 1e-12: # 避免除以零
            sinr_k = signal_power / 1e-12
        else:
            sinr_k = signal_power / (interference_power + noise_variance)

        if np.isnan(sinr_k) or np.isinf(sinr_k) or sinr_k < 0:
             rate_k = 0.0
        else:
            rate_k = np.log2(1 + sinr_k)
        sum_rate_calc += rate_k
        
    perf_metrics['sum_rate'] = sum_rate_calc
    return perf_metrics

# --- 绘图函数 ---
def plot_results(results: dict, params: SimulationParameters):
    """
    根据params.plot_metric绘制仿真结果。
    """
    num_schemes = len(params.precoding_schemes)
    markers = ['-o', '-s', '-^', '-d', '-*', '-x'] # 如果方案多于6个，则添加更多
    # colors = plt.cm.lines(np.linspace(0, 1, num_schemes)) # Matplotlib 默认颜色
    
    plt.figure(figsize=(10, 7))
    
    plot_type = params.plot_metric.lower()
    
    y_data_field = ''
    y_label_str = ''
    x_label_str = 'SNR (dB)'
    y_scale = 'linear'
    x_scale = 'linear'
    plot_title = ''

    if plot_type in ['sumrate', 'spectralefficiency']:
        y_data_field = 'sum_rate'
        if plot_type == 'spectralefficiency':
            y_label_str = '频谱效率 (bits/s/Hz)'
            plot_title = '频谱效率性能'
        else:
            y_label_str = '和速率 / 容量 (bits/s/Hz)'
            plot_title = '和速率性能'
    elif plot_type == 'ber':
        y_data_field = 'ber'
        y_label_str = '误比特率 (BER)'
        y_scale = 'log'
        plot_title = 'BER性能'
    elif plot_type == 'multiplexinggain':
        y_data_field = 'sum_rate'
        y_label_str = '和速率 / 容量 (bits/s/Hz)'
        plot_title = '和速率 vs SNR (观察复用增益)'
    elif plot_type == 'diversitygain':
        y_data_field = 'ber'
        y_label_str = '误比特率 (BER)'
        x_label_str = 'SNR (线性尺度)'
        y_scale = 'log'
        x_scale = 'log'
        plot_title = 'BER vs 线性SNR (观察分集增益)'
    else:
        raise ValueError(f'未知的绘图指标: {params.plot_metric}')

    for scheme_idx, scheme_name in enumerate(params.precoding_schemes):
        marker_style = markers[scheme_idx % len(markers)]
        y_data = np.array(results[scheme_name][y_data_field])

        if plot_type == 'diversitygain':
            x_data = 10**(params.SNR_dB_vec / 10) # 线性SNR用于loglog分集图
        else:
            x_data = params.SNR_dB_vec # dB SNR用于其他图

        if y_scale == 'log':
            y_data_plot = np.array([val if val > 0 else np.nan for val in y_data]) # 替换0为NaN以进行对数绘图
        else:
            y_data_plot = y_data
        
        plt.plot(x_data, y_data_plot, marker_style, linewidth=1.5, markersize=6, label=scheme_name)

    plt.grid(True, which="both", ls="--")
    plt.xlabel(x_label_str)
    plt.ylabel(y_label_str)
    plt.yscale(y_scale)
    plt.xscale(x_scale)

    if plot_type == 'ber' or plot_type == 'diversitygain':
        all_ber_positive = []
        for scheme_name in params.precoding_schemes:
            current_ber = np.array(results[scheme_name]['ber'])
            all_ber_positive.extend(current_ber[current_ber > 0])
        
        min_ber_overall = 1.0
        if all_ber_positive:
            min_ber_overall = min(all_ber_positive)
        else:
            min_ber_overall = 1e-7 # 如果没有错误，则为默认值
        
        plt.ylim(bottom=max(min_ber_overall * 0.1, 1e-7), top=1.1)


    title_full_str = f'{plot_title} (Nt={params.Nt}, K={params.K}, {params.channel_model} Ch'
    if params.channel_model.lower() == 'rician':
        title_full_str += f', K={params.rician_K_factor:.1f}'
    if plot_type == 'ber' or plot_type == 'diversitygain':
        title_full_str += f', {params.mod_order}-QAM'
    title_full_str += ')'
    plt.title(title_full_str)
    plt.legend(loc='best')
    plt.show()


# --- 主仿真循环 ---
if __name__ == '__main__':
    params = SimulationParameters()
    
    # --- 加载LCGNET模型 ---
    loaded_lcgnet_models = {}
    if not os.path.exists(params.lcgnet_model_dir):
        os.makedirs(params.lcgnet_model_dir)
        print(f"模型目录 {params.lcgnet_model_dir} 已创建。请确保预训练模型在此目录中。")

    # 尝试加载 LCGNET_CG 模型 (Hybrid_ZF_MRT_LCGNET_CG 和 LCGNET_CG 都需要它)
    lcgnet_cg_needed = 'LCGNET_CG' in params.precoding_schemes or 'Hybrid_ZF_MRT_LCGNET_CG' in params.precoding_schemes
    if lcgnet_cg_needed:
        try:
            model_cg = CGDUN(num_layers=params.lcgnet_num_layers_cg, k_dim=params.K, dtype=params.lcgnet_dtype).to(params.lcgnet_device)
            model_cg.load_state_dict(torch.load(params.lcgnet_model_path_cg, map_location=params.lcgnet_device))
            model_cg.eval()
            loaded_lcgnet_models['LCGNET_CG'] = model_cg
            print(f"LCGNET_CG 模型从 {params.lcgnet_model_path_cg} 加载成功。")
        except FileNotFoundError:
            print(f"错误: LCGNET_CG 模型文件 {params.lcgnet_model_path_cg} 未找到。依赖此模型的方案将不可用。")
            # 后续会从 schemes 列表中移除依赖方案
        except Exception as e:
            print(f"错误: 加载LCGNET_CG模型失败: {e}。依赖此模型的方案将不可用。")

    if 'LCGNET_Richardson' in params.precoding_schemes:
        try:
            model_richardson = RichardsonDUN(num_layers=params.lcgnet_num_layers_richardson, k_dim=params.K, dtype=params.lcgnet_dtype).to(params.lcgnet_device)
            model_richardson.load_state_dict(torch.load(params.lcgnet_model_path_richardson, map_location=params.lcgnet_device))
            model_richardson.eval()
            loaded_lcgnet_models['LCGNET_RICHARDSON'] = model_richardson
            print(f"LCGNET_Richardson 模型从 {params.lcgnet_model_path_richardson} 加载成功。")
        except FileNotFoundError:
            print(f"错误: LCGNET_Richardson 模型文件 {params.lcgnet_model_path_richardson} 未找到。该方案将不可用。")
        except Exception as e:
            print(f"错误: 加载LCGNET_Richardson模型失败: {e}。该方案将不可用。")
    # --- LCGNET模型加载结束 ---

    # --- 根据模型加载情况更新可用方案列表 ---
    active_precoding_schemes = []
    for scheme in params.precoding_schemes:
        if scheme == 'LCGNET_CG' and 'LCGNET_CG' not in loaded_lcgnet_models:
            print(f"方案 LCGNET_CG 已移除，因为其模型加载失败。")
            continue
        if scheme == 'LCGNET_Richardson' and 'LCGNET_RICHARDSON' not in loaded_lcgnet_models:
            print(f"方案 LCGNET_Richardson 已移除，因为其模型加载失败。")
            continue
        if scheme == 'Hybrid_ZF_MRT_LCGNET_CG' and 'LCGNET_CG' not in loaded_lcgnet_models:
            print(f"方案 Hybrid_ZF_MRT_LCGNET_CG 已移除，因为其依赖的 LCGNET_CG 模型加载失败。")
            continue
        active_precoding_schemes.append(scheme)
    params.precoding_schemes = active_precoding_schemes
    num_schemes = len(params.precoding_schemes)
    # --- 加载模型结束 ---

    # ---初始化结果结构---
    num_snr_points = len(params.SNR_dB_vec)
    num_schemes = len(params.precoding_schemes)
    snr_linear_vec = 10**(params.SNR_dB_vec / 10) # 线性SNR值
    # 使用字典存储结果
    results = {}
    print(f'初始化结果结构: {", ".join(params.precoding_schemes)}')
    for scheme_name in params.precoding_schemes:
        results[scheme_name] = {
            'sum_rate': np.zeros(num_snr_points),
            'ber': np.zeros(num_snr_points),
            'total_bits_simulated': np.zeros(num_snr_points),
            'total_errors_counted': np.zeros(num_snr_points),
            'total_tx_power': np.zeros(num_snr_points) # 新增
        }
        if scheme_name.upper().startswith('HYBRID'): # 检查是否为混合方案
             results[scheme_name]['optimized_lambda'] = np.zeros((params.K, params.num_monte_carlo, num_snr_points))
    print('初始化完成。\n')

    print('开始高级MIMO预编码仿真...')
    print(f'信道模型: {params.channel_model}')
    print(f'调制阶数: {params.mod_order}-QAM')
    print(f'预编码方案: {", ".join(params.precoding_schemes)}')
    start_time = time.time()

    for snr_idx, snr_db in enumerate(params.SNR_dB_vec):
        snr_linear = snr_linear_vec[snr_idx]
        # noise_variance = params.total_power / snr_linear # 噪声方差定义为 P_total / SNR
        # 在许多文献中，SNR定义为 E_s/N0，其中E_s是符号能量，N0是噪声功率谱密度。
        # 如果我们假设符号能量为1（归一化星座），那么噪声方差 sigma^2 = N0。
        # 如果总功率 P_total 分配给 K 个流，每个流的平均功率是 P_total/K (如果平均分配)。
        # 这里，我们假设噪声方差是相对于接收端每个天线（或每个用户，因为用户是单天线）的。
        # 如果 SNR = P_signal / P_noise，并且 P_signal 是在应用预编码器 *之后* 在接收机处测量的信号功率，
        # 而 P_noise 是噪声方差 sigma^2，那么这个定义是合理的。
        # MATLAB 代码中的 noise_variance = params.total_power / snr_linear 意味着
        # snr_linear = params.total_power / noise_variance。
        # 如果 params.total_power 是归一化的 (例如=1)，那么 noise_variance = 1 / snr_linear。
        # 这通常用于假设每个符号的平均接收功率（无衰落）为1。
        noise_variance = 1.0 / snr_linear # 假设归一化信号功率为1，则噪声方差为1/SNR

        print(f'  仿真 SNR = {snr_db} dB...')

        bits_simulated_snr = np.zeros(num_schemes)
        errors_counted_snr = np.zeros(num_schemes)
        sum_rate_acc_snr = np.zeros(num_schemes)
        actual_tx_power_acc_snr = np.zeros(num_schemes) # 新增

        for mc_run in range(params.num_monte_carlo):
            print(f'    MC 运行 {mc_run + 1}/{params.num_monte_carlo}')
            H = generate_channel_model(params) # 大小: K x Nt

            for scheme_idx, precoder_type in enumerate(params.precoding_schemes):
                W, lambda_opt_vec = apply_precoding(H, noise_variance, precoder_type, params, loaded_lcgnet_models)
                # 存储优化后的lambda值
                if precoder_type.upper().startswith('HYBRID') and not np.all(np.isnan(lambda_opt_vec)):
                     results[precoder_type]['optimized_lambda'][:, mc_run, snr_idx] = lambda_opt_vec
                # 计算功率，进行功率一致化判断
                if W is not None and np.all(np.isfinite(W)):
                    actual_power_W = np.real(np.trace(W @ W.conj().T))
                    actual_tx_power_acc_snr[scheme_idx] += actual_power_W
                else:
                    actual_tx_power_acc_snr[scheme_idx] += np.nan
                # 计算误码率
                if params.plot_ber.lower() == 'true':
                    tx_bits = np.random.randint(0, 2, size=(params.K * params.bits_per_symbol, params.num_symbols_per_run))
                    ber_metrics_val = calculate_ber_metrics(H, W, noise_variance, tx_bits, params)
                    errors_counted_snr[scheme_idx] += ber_metrics_val['error_count']
                    bits_simulated_snr[scheme_idx] += ber_metrics_val['bits_processed']
                #
                if params.plot_sumrate.lower() == 'true':
                    sumrate_metrics_val = calculate_sumrate_metrics(H, W, noise_variance, params)
                    sum_rate_acc_snr[scheme_idx] += sumrate_metrics_val['sum_rate']
        
        # 平均蒙特卡洛运行结果并存储
        for scheme_idx, scheme_name in enumerate(params.precoding_schemes):
            results[scheme_name]['sum_rate'][snr_idx] = sum_rate_acc_snr[scheme_idx] / params.num_monte_carlo
            
            if bits_simulated_snr[scheme_idx] > 0:
                results[scheme_name]['ber'][snr_idx] = errors_counted_snr[scheme_idx] / bits_simulated_snr[scheme_idx]
            else:
                results[scheme_name]['ber'][snr_idx] = np.nan # 避免除以零
            
            results[scheme_name]['total_bits_simulated'][snr_idx] = bits_simulated_snr[scheme_idx]
            results[scheme_name]['total_errors_counted'][snr_idx] = errors_counted_snr[scheme_idx]
            results[scheme_name]['total_tx_power'][snr_idx] = actual_tx_power_acc_snr[scheme_idx] / params.num_monte_carlo # 新增

            if params.plot_ber.lower() == 'true':
                print(f"    {scheme_name}: 错误数={errors_counted_snr[scheme_idx]}, 比特数={bits_simulated_snr[scheme_idx]}, BER={results[scheme_name]['ber'][snr_idx]:.2e}")
            if params.plot_sumrate.lower() == 'true':
                 print(f"    {scheme_name}: 平均和速率={results[scheme_name]['sum_rate'][snr_idx]:.4f}, 平均实际Tx功率={results[scheme_name]['total_tx_power'][snr_idx]:.4f}")


    end_time = time.time()
    print(f'仿真完成。耗时: {end_time - start_time:.2f} 秒。\n')

    # 结果可视化
    plot_results(results, params)
