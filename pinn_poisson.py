import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# ================== 1. 定义神经网络 ==================
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        # 结构: 2 -> 20 -> 20 -> 20 -> 20 -> 20 -> 20 -> 10 -> 1
        self.fc1 = nn.Linear(2, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 20)
        self.fc5 = nn.Linear(20, 20)
        self.fc6 = nn.Linear(20, 20)
        self.fc7 = nn.Linear(20, 10)   # 额外的一层
        self.fc8 = nn.Linear(10, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        # x shape: (N, 2)
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.act(self.fc3(out))
        out = self.act(self.fc4(out))
        out = self.act(self.fc5(out))
        out = self.act(self.fc6(out))
        out = self.act(self.fc7(out))
        out = self.fc8(out)   # 输出层无激活
        return out

# ================== 2. 定义源项和精确解 ==================
def f_exact(x, y):
    """源项 f(x,y) = (x^2 + y^2) * exp(x*y)"""
    return (x**2 + y**2) * torch.exp(x * y)

def u_exact(x, y):
    """精确解 exp(x*y)"""
    return torch.exp(x * y)

# ================== 3. 计算二阶导数（利用自动微分）==================
def compute_derivatives(model, x, y):
    """
    计算 u_xx, u_yy
    输入: x, y 形状 (N,)
    输出: u_xx, u_yy 形状 (N,)
    """
    # 需要梯度追踪
    x.requires_grad_(True)
    y.requires_grad_(True)
    # 将 x,y 拼接为 (N,2)
    xy = torch.stack([x, y], dim=1)
    u = model(xy)   # (N,1)
    # 计算 u_x, u_y
    u_x = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # 计算 u_xx, u_yy
    u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    return u_xx, u_yy

# ================== 4. 生成训练点 ==================
N_c = 400   # 内部配点数
N_b = 120   # 边界点数 (每条边 30)

# 内部随机点 (均匀采样)
x_c = torch.rand(N_c, 1) * 1.0   # (0,1)
y_c = torch.rand(N_c, 1) * 1.0

# 边界点：每条边 30 个点 (包括端点，但避免重复)
n_side = N_b // 4  # 30
t = torch.linspace(0, 1, n_side)  # 每条边上的参数 t

# 下边 y=0, x in [0,1]
x_b1 = t
y_b1 = torch.zeros_like(t)
# 上边 y=1
x_b2 = t
y_b2 = torch.ones_like(t)
# 左边 x=0, y in [0,1]
x_b3 = torch.zeros_like(t)
y_b3 = t
# 右边 x=1
x_b4 = torch.ones_like(t)
y_b4 = t

# 合并所有边界点
x_b = torch.cat([x_b1, x_b2, x_b3, x_b4], dim=0).unsqueeze(1)
y_b = torch.cat([y_b1, y_b2, y_b3, y_b4], dim=0).unsqueeze(1)

# 确保总数为 N_b
assert x_b.shape[0] == N_b

# ================== 5. 初始化模型和优化器 ==================
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# ================== 6. 训练循环 ==================
epochs = 60000   # 训练轮数
loss_history = {'pde': [], 'bc': [], 'total': []}

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    # 内部点预测和残差
    x_c.requires_grad_(True)
    y_c.requires_grad_(True)
    u_xx, u_yy = compute_derivatives(model, x_c.squeeze(), y_c.squeeze())
    # 计算残差
    residual = u_xx + u_yy - f_exact(x_c.squeeze(), y_c.squeeze())
    loss_pde = torch.mean(residual**2)
    
    # 边界损失
    x_b.requires_grad_(True)
    y_b.requires_grad_(True)
    u_pred_b = model(torch.cat([x_b, y_b], dim=1))
    u_true_b = u_exact(x_b.squeeze(), y_b.squeeze())
    loss_bc = torch.mean((u_pred_b.squeeze() - u_true_b)**2)
    
    # 总损失
    loss = loss_pde + loss_bc
    
    loss.backward()
    optimizer.step()
    
    # 记录损失
    if epoch % 100 == 0:
        loss_history['pde'].append(loss_pde.item())
        loss_history['bc'].append(loss_bc.item())
        loss_history['total'].append(loss.item())
        if epoch % 5000 == 0:
            print(f'Epoch {epoch:5d}, Loss PDE: {loss_pde.item():.2e}, BC: {loss_bc.item():.2e}, Total: {loss.item():.2e}')

print("训练完成！")

# ================== 7. 在测试网格上评估 ==================
n_grid = 100
x_grid = torch.linspace(0, 1, n_grid)
y_grid = torch.linspace(0, 1, n_grid)
X, Y = torch.meshgrid(x_grid, y_grid, indexing='ij')
x_flat = X.flatten()
y_flat = Y.flatten()

# 预测值
model.eval()
with torch.no_grad():
    xy_flat = torch.stack([x_flat, y_flat], dim=1)
    u_pred_flat = model(xy_flat).squeeze()
    u_exact_flat = u_exact(x_flat, y_flat)

# 重构成二维
u_pred = u_pred_flat.reshape(n_grid, n_grid)
u_exact_grid = u_exact_flat.reshape(n_grid, n_grid)
error = torch.abs(u_pred - u_exact_grid)

E_inf = torch.max(error).item()
E_mean = torch.mean(error).item()

print(f"E_inf = {E_inf:.4e}, E_mean = {E_mean:.4e}")

# ================== 8. 绘图 ==================
plt.figure(figsize=(16, 12))

# (1) 训练点分布 - 需 detach()
plt.subplot(3, 3, 1)
plt.scatter(x_c.detach().numpy(), y_c.detach().numpy(), s=10, label='Interior collocation', alpha=0.6)
plt.scatter(x_b.detach().numpy(), y_b.detach().numpy(), s=10, c='r', label='Boundary points')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Training Points')
plt.legend(); plt.axis('equal')

# (2) 损失历史 (对数)
plt.subplot(3, 3, 2)
iters = np.arange(0, epochs, 100)
plt.semilogy(iters, loss_history['pde'], label='PDE Loss')
plt.semilogy(iters, loss_history['bc'], label='BC Loss')
plt.semilogy(iters, loss_history['total'], label='Total Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss (log)'); plt.title('Loss History')
plt.legend(); plt.grid(True)

# (3) PINN 预测解 - detach().numpy()
plt.subplot(3, 3, 3)
plt.contourf(X.numpy(), Y.numpy(), u_pred.detach().numpy(), levels=100, cmap='viridis')
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y'); plt.title('PINN Prediction')

# (4) 精确解 - 直接 .numpy() 即可（无梯度）
plt.subplot(3, 3, 4)
plt.contourf(X.numpy(), Y.numpy(), u_exact_grid.numpy(), levels=100, cmap='viridis')
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y'); plt.title('Exact Solution')

# (5) 绝对误差 - detach()
plt.subplot(3, 3, 5)
plt.contourf(X.numpy(), Y.numpy(), error.detach().numpy(), levels=100, cmap='hot')
plt.colorbar(); plt.xlabel('x'); plt.ylabel('y'); plt.title('Absolute Error')

# (6) 沿 y=0.5 的对比
plt.subplot(3, 3, 6)
x_line = torch.linspace(0, 1, 200)
y_line = 0.5 * torch.ones_like(x_line)
with torch.no_grad():
    xy_line = torch.stack([x_line, y_line], dim=1)
    u_pred_line = model(xy_line).squeeze()
    u_exact_line = u_exact(x_line, y_line)
plt.plot(x_line.numpy(), u_pred_line.detach().numpy(), 'b-', label='PINN')
plt.plot(x_line.numpy(), u_exact_line.numpy(), 'r--', label='Exact')
plt.xlabel('x'); plt.ylabel('u(x,0.5)'); plt.title('Comparison at y=0.5')
plt.legend(); plt.grid(True)

plt.tight_layout()
plt.show()