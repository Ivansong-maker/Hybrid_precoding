
# gemini生成的代码，根据项目框架
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt

# 模型保存目录
MODEL_SAVE_DIR = "/models"

# --- Configuration ---
K = 32  # Number of users (also dimension of HH^H)
Nt = 128 # Number of transmit antennas
NUM_LAYERS_RICHARDSON = 10 # Number of layers (iterations) for Richardson DUN
NUM_LAYERS_CG = 5       # Number of layers (iterations) for CG DUN (CG typically converges faster)
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
NUM_EPOCHS = 500 # Adjust as needed
NUM_TRAIN_SAMPLES = BATCH_SIZE * 100
NUM_VAL_SAMPLES = BATCH_SIZE * 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.complex64 # Data type for complex numbers

# --- Helper Functions ---
def generate_channel_h(num_samples, k, nt, dtype=DTYPE, device=DEVICE):
    """Generates random MIMO channel matrices H."""
    # For simplicity, using i.i.d. Rayleigh fading channel
    # H_real = torch.randn(num_samples, k, nt, device=device)
    # H_imag = torch.randn(num_samples, k, nt, device=device)
    # H = (1/np.sqrt(2)) * (H_real + 1j * H_imag).to(dtype)

    # More direct way for complex normal
    H = (torch.randn(num_samples, k, nt, dtype=dtype, device=device) + \
         1j * torch.randn(num_samples, k, nt, dtype=dtype, device=device)) / np.sqrt(2.0)
    return H

def calculate_A_and_true_inverse(H_batch):
    """Calculates A = HH^H and its true inverse for a batch of H."""
    A_batch = torch.matmul(H_batch, H_batch.conj().transpose(-2, -1))
    
    # Add a small identity matrix for numerical stability before inversion,
    # especially if A_batch might be ill-conditioned or not perfectly full rank.
    # This is a practical consideration even if the ideal formula doesn't have it.
    # For KxK matrix, K=32, direct inversion is feasible for generating labels.
    # If HH^H is singular, inv() will fail. This might happen if K > Nt or rows are linearly dependent.
    # Given K=32, Nt=128, HH^H should generally be invertible.
    # eye_A = torch.eye(A_batch.size(-1), dtype=A_batch.dtype, device=A_batch.device)
    # A_inv_batch_true = torch.linalg.inv(A_batch + 1e-9 * eye_A) # Small regularization for stability
    
    A_inv_batch_true = []
    for i in range(A_batch.shape[0]):
        try:
            # Attempt direct inversion
            A_inv_batch_true.append(torch.linalg.inv(A_batch[i]))
        except torch.linalg.LinAlgError:
            # If singular, use pseudo-inverse as a fallback for label generation
            # Or add stronger regularization for this specific sample
            print(f"Warning: Matrix at index {i} in batch might be singular. Using pseudo-inverse for label.")
            A_inv_batch_true.append(torch.linalg.pinv(A_batch[i])) # Pseudo-inverse

    A_inv_batch_true = torch.stack(A_inv_batch_true)
    return A_batch, A_inv_batch_true

# --- DUN based on Richardson Iteration ---
class RichardsonDUN(nn.Module):
    def __init__(self, num_layers, k_dim, dtype=DTYPE):
        super(RichardsonDUN, self).__init__()
        self.num_layers = num_layers
        self.k_dim = k_dim
        self.dtype = dtype

        # Trainable step sizes (omegas) for each layer
        # Initialize them with small random values or a constant
        # Using nn.ParameterList to correctly register parameters
        self.omegas = nn.ParameterList([nn.Parameter(torch.rand(1, dtype=torch.float32) * 0.1) 
                                          for _ in range(num_layers)])
        
        # Optional: Trainable initial scaling factor c0 for X0 = c0 * I
        # self.c0 = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        # For simplicity, we'll use X0 = 0 or a fixed small identity matrix.

    def forward(self, A_batch):
        batch_size = A_batch.shape[0]
        
        # Initialize X0
        # X_approx = self.c0 * torch.eye(self.k_dim, dtype=self.dtype, device=A_batch.device).unsqueeze(0).repeat(batch_size, 1, 1)
        X_approx = torch.zeros(batch_size, self.k_dim, self.k_dim, dtype=self.dtype, device=A_batch.device)
        # Or:
        # X_approx = 0.01 * torch.eye(self.k_dim, dtype=self.dtype, device=A_batch.device).unsqueeze(0).repeat(batch_size, 1, 1)


        I_k = torch.eye(self.k_dim, dtype=self.dtype, device=A_batch.device).unsqueeze(0).repeat(batch_size, 1, 1)

        for i in range(self.num_layers):
            omega = self.omegas[i].to(self.dtype) # Ensure omega is complex if X and A are complex
            residual_term = I_k - torch.matmul(A_batch, X_approx)
            X_approx = X_approx + omega * residual_term
            
        return X_approx

# --- DUN based on Conjugate Gradient (CG) Method (Unfolded) ---
class CGDUN(nn.Module):
    def __init__(self, num_layers, k_dim, dtype=DTYPE):
        super(CGDUN, self).__init__()
        self.num_layers = num_layers
        self.k_dim = k_dim
        self.dtype = dtype

        # Trainable parameters for each layer
        # For CG: alpha_k and beta_k (or parameters to compute them)
        # Simplified approach: make alpha and beta directly trainable per layer
        self.alphas = nn.ParameterList([nn.Parameter(torch.rand(1, dtype=torch.float32) * 0.1) 
                                          for _ in range(num_layers)])
        self.betas = nn.ParameterList([nn.Parameter(torch.rand(1, dtype=torch.float32) * 0.01) 
                                         for _ in range(num_layers)])
        
        # Optional: Trainable initial scaling factor c0 for X0
        # self.c0 = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))


    def forward(self, A_batch):
        batch_size = A_batch.shape[0]
        
        # Initialize X0 (solution estimate)
        # X_approx = self.c0 * torch.eye(self.k_dim, dtype=self.dtype, device=A_batch.device).unsqueeze(0).repeat(batch_size, 1, 1)
        X_approx = torch.zeros(batch_size, self.k_dim, self.k_dim, dtype=self.dtype, device=A_batch.device)

        I_k = torch.eye(self.k_dim, dtype=self.dtype, device=A_batch.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # Initialize r0 (residual) and p0 (search direction)
        # For AX=I, initial residual r0 = I - AX0. If X0=0, r0=I.
        r_current = I_k - torch.matmul(A_batch, X_approx) 
        p_current = r_current.clone() # p0 = r0

        for i in range(self.num_layers):
            alpha_param = self.alphas[i].to(self.dtype)
            beta_param = self.betas[i].to(self.dtype)

            # Standard CG formulas:
            # Ap_k = A * p_k
            # alpha_k_cg = (r_k^H * r_k) / (p_k^H * Ap_k)  (sum over matrix elements for Frobenius norm like behavior)
            # X_{k+1} = X_k + alpha_k_cg * p_k
            # r_{k+1} = r_k - alpha_k_cg * Ap_k
            # beta_k_cg = (r_{k+1}^H * r_{k+1}) / (r_k^H * r_k)
            # p_{k+1} = r_{k+1} + beta_k_cg * p_k

            # Unfolded version: use trainable alpha_param and beta_param
            Ap = torch.matmul(A_batch, p_current)
            
            # Update solution
            X_approx = X_approx + alpha_param * p_current
            
            # Update residual
            r_next = r_current - alpha_param * Ap
            
            # Update search direction
            # Note: In standard CG for Ax=b where x,b are vectors, beta is scalar.
            # Here X, R, P are matrices. We use scalar alpha, beta for simplicity.
            p_current = r_next + beta_param * p_current # Element-wise scaling by beta
            
            r_current = r_next
            
        return X_approx

# --- Loss Functions ---
def direct_mse_loss(X_approx, X_true):
    """Computes ||X_approx - X_true||_F^2 / N_batch"""
    loss = torch.mean(torch.sum(torch.abs(X_approx - X_true)**2, dim=(-2, -1)))
    return loss

def residual_loss(A_batch, X_approx, I_k_batch):
    """Computes ||A * X_approx - I||_F^2 / N_batch"""
    product = torch.matmul(A_batch, X_approx)
    loss = torch.mean(torch.sum(torch.abs(product - I_k_batch)**2, dim=(-2, -1)))
    return loss

# --- Training Loop ---
def train_model(model, train_H_loader, val_H_loader, optimizer, num_epochs, loss_type="mse", device=DEVICE, model_save_path=None):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    k_dim = model.k_dim
    I_k_batch_template = torch.eye(k_dim, dtype=DTYPE, device=device) # For residual loss

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        start_time = time.time()

        for H_batch_train, _ in train_H_loader:  # Unpack the tuple and ignore the dummy target
            H_batch_train = H_batch_train.to(device)
            A_batch_train, A_inv_batch_train_true = calculate_A_and_true_inverse(H_batch_train)

            optimizer.zero_grad()
            X_approx_train = model(A_batch_train)

            if loss_type == "mse":
                loss = direct_mse_loss(X_approx_train, A_inv_batch_train_true)
            elif loss_type == "residual":
                I_k_batch = I_k_batch_template.unsqueeze(0).repeat(A_batch_train.shape[0], 1, 1)
                loss = residual_loss(A_batch_train, X_approx_train, I_k_batch)
            else:
                raise ValueError("Unknown loss_type")

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * H_batch_train.size(0)

        epoch_train_loss /= len(train_H_loader.dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for H_batch_val, _ in val_H_loader:  # Unpack the tuple and ignore the dummy target
                H_batch_val = H_batch_val.to(device)
                A_batch_val, A_inv_batch_val_true = calculate_A_and_true_inverse(H_batch_val)
                X_approx_val = model(A_batch_val)

                if loss_type == "mse":
                    val_loss_item = direct_mse_loss(X_approx_val, A_inv_batch_val_true)
                elif loss_type == "residual":
                    I_k_batch = I_k_batch_template.unsqueeze(0).repeat(A_batch_val.shape[0], 1, 1)
                    val_loss_item = residual_loss(A_batch_val, X_approx_val, I_k_batch)
                epoch_val_loss += val_loss_item.item() * H_batch_val.size(0)
        
        epoch_val_loss /= len(val_H_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.4e}, Val Loss: {epoch_val_loss:.4e}, Time: {epoch_time:.2f}s")
        if model_save_path and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"   Best model saved to {model_save_path} (Val Loss: {best_val_loss:.4e})")

    return train_losses, val_losses
def test_trained_model(model, num_test_samples=3, k_dim=K, nt_dim=Nt, device=DEVICE, dtype=DTYPE):
    """
    测试训练好的模型，并直观展示其对矩阵逆的拟合效果。
    """
    model.eval() #确保模型处于评估模式
    print("\n--- Testing Trained Model ---")

    # 生成一些测试用的信道矩阵 H
    H_test = generate_channel_h(num_test_samples, k_dim, nt_dim, dtype=dtype, device=device)
    A_test, A_inv_true_test = calculate_A_and_true_inverse(H_test)

    with torch.no_grad(): # 测试时不需要计算梯度
        A_inv_approx_test = model(A_test)

    # 逐个样本比较结果
    for i in range(num_test_samples):
        print(f"\n--- Test Sample {i+1} ---")
        A_i = A_test[i]
        A_inv_true_i = A_inv_true_test[i]
        A_inv_approx_i = A_inv_approx_test[i]

        # 1. 打印真实逆矩阵和近似逆矩阵 (为简洁，可只打印部分元素或范数)
        # print(f"A_inv_true (sample {i+1}, first 3x3 block):\n{A_inv_true_i[:3,:3].cpu().numpy()}")
        # print(f"A_inv_approx (sample {i+1}, first 3x3 block):\n{A_inv_approx_i[:3,:3].cpu().numpy()}")

        # 2. 计算近似误差的 Frobenius 范数
        diff_norm = torch.norm(A_inv_true_i - A_inv_approx_i, p='fro')
        true_norm = torch.norm(A_inv_true_i, p='fro')
        relative_error = diff_norm / true_norm if true_norm > 1e-9 else diff_norm # 避免除以零
        print(f"Frobenius Norm of (A_inv_true - A_inv_approx): {diff_norm.item():.4e}")
        print(f"Relative Frobenius Error: {relative_error.item():.4%}")

        # 3. 验证 A * A_inv_approx 是否接近单位矩阵 I
        identity_approx = torch.matmul(A_i, A_inv_approx_i)
        identity_true = torch.eye(k_dim, dtype=dtype, device=device)
        identity_diff_norm = torch.norm(identity_approx - identity_true, p='fro')
        print(f"Frobenius Norm of (A * A_inv_approx - I): {identity_diff_norm.item():.4e}")

        # (可选) 打印 A * A_inv_approx 的部分元素
        # print(f"A * A_inv_approx (sample {i+1}, first 3x3 block):\n{identity_approx[:3,:3].cpu().numpy()}")
        # print(f"Identity (first 3x3 block for reference):\n{identity_true[:3,:3].cpu().numpy()}")

    print("\n--- End of Testing ---")


# --- Main Execution (修改后) ---
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    # 创建模型保存目录
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    # 生成数据 (如果需要重新生成)
    print("Generating training and validation data...")
    H_train_data = generate_channel_h(NUM_TRAIN_SAMPLES, K, Nt, DTYPE, DEVICE)
    H_val_data = generate_channel_h(NUM_VAL_SAMPLES, K, Nt, DTYPE, DEVICE)

    # Create datasets that return tuples (H_data, None) to match the expected format
    train_dataset = torch.utils.data.TensorDataset(H_train_data, torch.zeros(len(H_train_data)))
    val_dataset = torch.utils.data.TensorDataset(H_val_data, torch.zeros(len(H_val_data)))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Data generation complete.")

    # --- 选择模型、损失函数并进行训练 ---
    model_type = "CG"  # 可选 "Richardson" 或 "CG"
    selected_loss_type = "mse" # 可选 "mse" 或 "residual"
    model_filename = f"{model_type}_dun_{selected_loss_type}_k{K}_nt{Nt}_l{NUM_LAYERS_CG if model_type == 'CG' else NUM_LAYERS_RICHARDSON}.pth"
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)

    print(f"\nTraining {model_type}-DUN with {selected_loss_type} loss...")
    print(f"Model will be saved to: {model_save_path}")

    if model_type == "Richardson":
        dun_model = RichardsonDUN(num_layers=NUM_LAYERS_RICHARDSON, k_dim=K, dtype=DTYPE).to(DEVICE)
    elif model_type == "CG":
        dun_model = CGDUN(num_layers=NUM_LAYERS_CG, k_dim=K, dtype=DTYPE).to(DEVICE)
    else:
        raise ValueError("Invalid model_type specified.")

    optimizer = optim.Adam(dun_model.parameters(), lr=LEARNING_RATE)

    # 训练模型，并在验证集上表现最好时保存
    train_history, val_history = train_model(dun_model,
                                             train_loader,
                                             val_loader,
                                             optimizer,
                                             NUM_EPOCHS, # 实际训练时使用配置中的 NUM_EPOCHS
                                             loss_type=selected_loss_type,
                                             device=DEVICE,
                                             model_save_path=model_save_path) # 传递保存路径

    print("Training complete.")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_history) + 1), train_history, label=f'Training Loss ({selected_loss_type.upper()})')
    plt.plot(range(1, len(val_history) + 1), val_history, label=f'Validation Loss ({selected_loss_type.upper()})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {model_type}-DUN')
    plt.legend()
    plt.grid(True)
    loss_curve_filename = f"{model_type}_dun_{selected_loss_type}_loss_curve.png"
    plt.savefig(loss_curve_filename)
    print(f"Loss curve saved to {loss_curve_filename}")
    # plt.show()

    # --- 加载训练好的模型并进行测试 ---
    print(f"\nLoading trained model from {model_save_path} for testing...")
    # 重新实例化模型结构
    if model_type == "Richardson":
        model_for_testing = RichardsonDUN(num_layers=NUM_LAYERS_RICHARDSON, k_dim=K, dtype=DTYPE).to(DEVICE)
    elif model_type == "CG":
        model_for_testing = CGDUN(num_layers=NUM_LAYERS_CG, k_dim=K, dtype=DTYPE).to(DEVICE)
    else: # Should not happen if training was successful
        raise ValueError("Invalid model_type for loading.")

    try:
        model_for_testing.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
        print("Model loaded successfully.")
        # 调用测试函数
        test_trained_model(model_for_testing, num_test_samples=5, k_dim=K, nt_dim=Nt, device=DEVICE, dtype=DTYPE)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_save_path}. Skipping testing.")
    except Exception as e:
        print(f"Error loading model or during testing: {e}. Skipping testing.")