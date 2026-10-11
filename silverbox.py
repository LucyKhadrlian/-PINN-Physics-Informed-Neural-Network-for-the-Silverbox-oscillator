import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch
import torch.nn as nn

# ---------------- 1. Load data ----------------
mat = scipy.io.loadmat(r"C:\Users\kevor\OneDrive\Desktop\Lucy_AI Training\hw_1\task 5\SilverboxFiles\SNLS80mV.mat")
u = mat["V1"].flatten().astype(np.float64)
y = mat["V2"].flatten().astype(np.float64)

fs = 610.35
dt = 1.0 / fs
start = 42650
N = 4096
u = u[start:start+N].copy()
y = y[start:start+N].copy()
u -= u.mean()
y -= y.mean()
N_train = 3072

v_fd = np.gradient(y, dt)                       # velocity estimate (finite difference)

# scales so the network sees numbers of order 1
Sy, Su = y[:N_train].std(), u[:N_train].std()
w0 = 2 * np.pi * 70
Sv, Sa = w0 * Sy, w0**2 * Sy

y_t = torch.tensor(y, dtype=torch.float32)
u_t = torch.tensor(u, dtype=torch.float32)
v_t = torch.tensor(v_fd, dtype=torch.float32)

# ---------------- 2. Acceleration network ----------------
net = nn.Sequential(nn.Linear(3, 32), nn.Tanh(),
                    nn.Linear(32, 32), nn.Tanh(),
                    nn.Linear(32, 1))
lin = nn.Linear(3, 1)                           # linear spring/damper/gain path

def accel(y, v, u):
    x = torch.stack([y / Sy, v / Sv, u / Su], dim=-1)
    return (net(x) + lin(x)).squeeze(-1) * Sa

# ---------------- 3. RK4 step ----------------
def rk4(y, v, u0, u1):
    um = 0.5 * (u0 + u1)
    k1y, k1v = v, accel(y, v, u0)
    k2y, k2v = v + 0.5*dt*k1v, accel(y + 0.5*dt*k1y, v + 0.5*dt*k1v, um)
    k3y, k3v = v + 0.5*dt*k2v, accel(y + 0.5*dt*k2y, v + 0.5*dt*k2v, um)
    k4y, k4v = v + dt*k3v, accel(y + dt*k3y, v + dt*k3v, u1)
    y_new = y + dt/6 * (k1y + 2*k2y + 2*k3y + k4y)
    v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
    return y_new, v_new

# ---------------- 3b. Pretrain on finite-difference acceleration ----------------
# (training record only, no integration, so it is fast)
a_fd = np.zeros(N)
a_fd[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / dt**2
a_t = torch.tensor(a_fd, dtype=torch.float32)
idx_tr = torch.arange(1, N_train - 1)

params = list(net.parameters()) + list(lin.parameters())
opt0 = torch.optim.Adam(params, lr=3e-3)
for it in range(3000):
    pred_a = accel(y_t[idx_tr], v_t[idx_tr], u_t[idx_tr])
    loss0 = torch.mean((pred_a - a_t[idx_tr])**2) / Sa**2
    opt0.zero_grad()
    loss0.backward()
    opt0.step()
    if it % 500 == 0:
        print(f"pretrain it={it} loss={loss0.item():.3e}", flush=True)

# ---------------- 4. Fine-tune on short rollouts ----------------
opt = torch.optim.Adam(params, lr=1e-3)

# first run: use [(16, 100), (64, 50)] to test quickly, then increase the steps
for L, steps in [(16, 300), (64, 200), (128, 100)]:      # rollout length, steps
    for it in range(steps):
        st = np.random.randint(0, N_train - L - 1, size=32)
        idx = torch.tensor(st[:, None] + np.arange(L + 1))
        ys, us = y_t[idx], u_t[idx]
        yy, vv = ys[:, 0], v_t[idx[:, 0]]

        preds = []
        for i in range(L):
            yy, vv = rk4(yy, vv, us[:, i], us[:, i+1])
            preds.append(yy)

        loss = torch.mean((torch.stack(preds, dim=1) - ys[:, 1:])**2) / Sy**2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 50 == 0:
            print(f"L={L} it={it} loss={loss.item():.3e}", flush=True)

# ---------------- 5. Free run over the whole excerpt ----------------
yy = torch.tensor([y[0]], dtype=torch.float32)
vv = torch.tensor([(y[1] - y[0]) / dt], dtype=torch.float32)
y_pred = [y[0]]
with torch.no_grad():
    for k in range(N - 1):
        yy, vv = rk4(yy, vv, u_t[k:k+1], u_t[k+1:k+2])
        y_pred.append(yy.item())
y_pred = np.array(y_pred)

rmse_train = np.sqrt(np.mean((y_pred[:N_train] - y[:N_train])**2))
rmse_test  = np.sqrt(np.mean((y_pred[N_train:] - y[N_train:])**2))
print(f"Free-run RMSE (train): {rmse_train:.6f} V")
print(f"Free-run RMSE (test) : {rmse_test:.6f} V")


# ---------------- 5b. Diagnostic: least-squares fit of eq. (1) + same RK4 ----------------
sl = slice(1, N_train - 1)
A = np.stack([-v_fd[sl], -y[sl], -y[sl]**3, u[sl]], axis=1)
coef = np.linalg.lstsq(A, a_fd[sl], rcond=None)[0]
a_c, b_c, c_c, g_c = coef
print("LS coefficients a, b, c, g =", coef, "| (2*pi*70)^2 =", w0**2)

def f_ls(y, v, u):
    return -a_c*v - b_c*y - c_c*y**3 + g_c*u

def rk4_np(y0, v0, u0, u1):
    um = 0.5 * (u0 + u1)
    k1y, k1v = v0, f_ls(y0, v0, u0)
    k2y, k2v = v0 + 0.5*dt*k1v, f_ls(y0 + 0.5*dt*k1y, v0 + 0.5*dt*k1v, um)
    k3y, k3v = v0 + 0.5*dt*k2v, f_ls(y0 + 0.5*dt*k2y, v0 + 0.5*dt*k2v, um)
    k4y, k4v = v0 + dt*k3v, f_ls(y0 + dt*k3y, v0 + dt*k3v, u1)
    return (y0 + dt/6*(k1y + 2*k2y + 2*k3y + k4y),
            v0 + dt/6*(k1v + 2*k2v + 2*k3v + k4v))

yy_ls, vv_ls = y[0], (y[1] - y[0]) / dt
y_ls = [yy_ls]
for k in range(N - 1):
    yy_ls, vv_ls = rk4_np(yy_ls, vv_ls, u[k], u[k+1])
    y_ls.append(yy_ls)
y_ls = np.array(y_ls)
print(f"LS-physics free-run RMSE (train): {np.sqrt(np.mean((y_ls[:N_train] - y[:N_train])**2)):.6f} V")
print(f"LS-physics free-run RMSE (test) : {np.sqrt(np.mean((y_ls[N_train:] - y[N_train:])**2)):.6f} V")

# ---------------- 6. Plot ----------------
t = np.arange(N - N_train) * dt
plt.figure(figsize=(11, 4))
plt.plot(t, y[N_train:], label="measured", lw=1.5)
plt.plot(t, y_pred[N_train:], label="predicted (free run)", lw=1.3, alpha=0.85)
plt.plot(t, y_ls[N_train:], label="least-squares physics model", lw=1.0, alpha=0.7, ls="--")
plt.xlabel("time [s]")
plt.ylabel("y [V]")
plt.title(f"Test free-run   RMSE = {rmse_test:.4f} V")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("test_free_run.png", dpi=150)
plt.show()