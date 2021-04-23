# ------------------------------------------------------------------------------
# @author  ingridn
# @brief   Hw6
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------
np.set_printoptions(suppress=True, precision=5)

# ------------------------------------------------------------------------------
# q2.c
print("\n#", "-"*50)
print("q2 c) Kalman Filter")
ydata = np.loadtxt("ydata.txt", delimiter=' ', dtype=np.float32)
print(f"\n*** Measurements:\n{ydata}")

# parameters
dt = 1
M = 0.01
u = 0.01 * np.ones(shape=(3, 1), dtype=np.float32)
P0 = np.diag([50, 50, 50, 10, 10, 10])
Rv = 1e-5 * np.diag([1, 1, 1])
Rw = 50 * np.diag([1, 1, 1])
print(f"\n*** Parameters:")
print(f"dt: {dt}\nM: {M}\nu:\n{u}\nP0:\n{P0}\nRv:\n{Rv}\nRw:\n{Rw}")

print("\n*** State Space Representation:")
A = np.identity(6, dtype=np.float32)
A[0, 3] = A[1, 4] = A[2, 5] = dt

B = np.zeros(shape=(6, 3), dtype=np.float32)
B[0, 0] = B[1, 1] = B[2, 2] = 0.5 * dt/M
B[3, 0] = B[4, 1] = B[5, 2] = dt/M

C = np.zeros(shape=(3, 6), dtype=np.float32)
C[0, 0] = C[1, 1] = C[2, 2] = 1
print(f"A:\n{A}\nB:\n{B}\nC:\n{C}")

# kalman filter 
print("\n*** Kalman filter:")
# initial covariance
Pprior = np.zeros(shape=(len(ydata)+1, 6, 6))
Pprior[0, :, :] = P0

Ppost = np.zeros(shape=(len(ydata)+1, 6, 6))
Ppost[0, :, :] = P0

# initial state
x = np.zeros(shape=(6, 1), dtype=np.float32)
x[0] = ydata[0, 0]
x[1] = ydata[0, 1]
x[2] = ydata[0, 2]

Xprior = np.zeros(shape=(len(ydata)+1, 6, 1))
Xprior[0, :, :] = x

Xpost = np.zeros(shape=(len(ydata)+1, 6, 1))
Xpost[0, :, :] = x

y_kf = np.zeros(shape=ydata.shape, dtype=np.float32)
I = np.identity(6)

luenberger = False
    
if luenberger:
    P_list = np.zeros(shape=(len(ydata)+1, 6, 6))
    P_list[0, :, :] = P0
   
    x_list = np.zeros(shape=(len(ydata)+1, 6, 1))
    x_list[0, :, :] = x

    y_kf = np.zeros(shape=ydata.shape, dtype=np.float32)
    for i, ym in enumerate(ydata):
        P = P_list[i]
        print(f"\nym[{i}]: {ym}")
        # compute observer gain L -- AM p. 215
        # L = A P_k C^{T} (R_w + C P_k C^{T})^{-1}
        L = A @ P @ C.T @ np.linalg.inv((Rw + C @ P @ C.T))
        print(f"observer gain L:\n{L}")
        # compute new covariance matrix P -- AM p. 215
        # P = (A - LC) P_k (A - LC)^{T} + F R_v F^{T} + L R_w L^{T}
        # here F = B
        A_LC = A - L @ C
        P = A_LC @ P @ A_LC.T + B @ Rv @ B.T + L @ Rw @ L.T
        P_list[i+1] = P
        print(f"covariance matrix P:\n{P}")
        # compute new x as a Luenberger observer
        # x = Ax^ + Bu + L(y - Cx^)
        # diff between observer prediction and measured output
        ym = ym.reshape(3, 1)
        err = ym - C @ x
        diff = L @ (err) 
        
        # estimate state
        x = A @ x + B @ u + diff
        x_list[i+1] = x
        print(f"x[{i}]:\n{x}")
        y_kf[i, :] = (C @ x).reshape(3)
        print(f"y[{i}]:\n{C @ x}")
        
        print(f"err:\n{err}")
else:
    for k, ym in enumerate(ydata):
        print(f"\n\n*** step[{k+1}/{len(ydata)}]")
        print(f"* y_m:\n{ym}")
        
        # priors state
        Xprior[k+1] = A @ Xpost[k] + B @ u
        print(f"* x_prior:\n{Xprior[k+1].T}")
        
        # prior covariance
        Pprior[k+1] = A @ Ppost[k] @ A.T + B @ Rv @ B.T 
        print(f"* P_prior:\n{Pprior[k+1]}")
        
        # kalman gain
        K = Pprior[k+1] @ C.T @ np.linalg.inv(C @ Pprior[k+1] @ C.T + Rw)
        print(f"* K\n{K}")
        
        # posterior state
        ym = ym.reshape(3, 1)
        Xpost[k+1] = Xprior[k+1] + K @ (ym - C @ Xprior[k+1])
        print(f"* x_post:\n{Xpost[k+1].T}")
        print(f"* err:\n{(ym - C @ Xprior[k+1]).T}")
        
        # posterior covariance
        Ppost[k+1] = (I - K @ C) @ Pprior[k+1] 
        print(f"* P_post:\n{Ppost[k+1]}")
        
        y_kf[k, :] = (C @ Xpost[k+1]).reshape(3)
        print(f"* y_kf:\n{y_kf[k]}")

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
t = np.arange(0, len(ydata), step=1)
ax.plot3D(ydata[:, 0], ydata[:, 1], ydata[:, 2], marker='o', markersize=3, 
          linestyle='dashed', linewidth=1, color='blue', label='y_noisy')
ax.plot3D(y_kf[:, 0], y_kf[:, 1], y_kf[:, 2], 
          marker='+', markersize=3, color='red', label='y_kf')
ax.legend()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.savefig("kalman_filter.png", bbox_inches='tight')
plt.show()
plt.close()

    
# ------------------------------------------------------------------------------
# q2.d
print("\n#", "-"*50)
print("q2 d) Rauch-Tung-Striebel smoother")

Ps = Ppost
Xs = Xpost

k = len(Ps) - 2
y_rts = np.zeros_like(ydata)
while k >= 0:
    print(f"\n\n*** step[{k+1}/{len(Ps)}]")
    # equations to compute 
    #   k < n
    #   x_{k|n} = x_{k|k} + C_{k} (x_{k+1|n} - x_{k+1|k})
    #   P_{k|n} = P_{k|k} + C_{k} (P_{k+1|n} - P_{k+1|k}) C_{k}^{T}
    #     C_{k} = P_{k|k} B^{T} P_{k+1|k}^{-1}
    # compute c
    Ck = Ppost[k] @ A.T @ np.linalg.inv(Pprior[k+1])
    
    Xs[k] = Xpost[k] + Ck @ (Xs[k+1] - Xprior[k+1])
    print(f"* xs:\n{Xs[k].T}")
    
    Ps[k] = Ppost[k] + Ck @ (Ps[k+1] - Pprior[k+1]) @ Ck.T
    print(f"* Ps:\n{Ps[k]}")
    
    y_rts[k] = (C @ Xs[k]).reshape(3)
    k -= 1

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
t = np.arange(0, len(ydata), step=1)
ax.plot3D(ydata[:, 0], ydata[:, 1], ydata[:, 2], marker='o', markersize=3, 
          linestyle='dashed', linewidth=1, color='blue', label='y_noisy')
ax.plot3D(y_kf[:, 0], y_kf[:, 1], y_kf[:, 2], marker='+', markersize=3, 
          linewidth=1, color='red', label='y_kf')
ax.plot3D(y_rts[:, 0], y_rts[:, 1], y_rts[:, 2], 
          marker='v', markersize=3, color='green', label='y_rts')
ax.legend()
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.savefig("kalman_rts.png", bbox_inches='tight')
plt.show()
plt.close()
