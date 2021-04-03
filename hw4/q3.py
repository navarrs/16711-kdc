import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion
from utils import Skew, Tilde
from scipy.spatial.transform import Rotation

poses = np.loadtxt('poses.txt')
T = poses[:, 0]
X = poses[:, 1]
Y = poses[:, 2]
Z = poses[:, 3]

quats = poses[:, 4:]

# q3.1 - velocity
points = poses[:, 1:4]
N = T.shape[0]
A = np.zeros(shape=(3*N, 9), dtype=np.float)
b = np.zeros(shape=(3*N), dtype=np.float)
I = np.identity(3)


def ComputeWb(q1, q2):
    
    qd = q2 * q1.inverse
    theta = 2 * math.acos(qd.real)
    ws = np.zeros(shape=3)
    if not np.isclose(theta, 0):
        ws = qd.imaginary / math.sin(theta/2)
        ws = theta * ws / np.linalg.norm(ws)
    
    return q1.rotation_matrix.T @ ws

# q3.1 - velocity of center of mass 
# q3.4 - distance from center of mass to beacon
wb = []
for t in range(N):
    xyz = points[t]

    q = quats[t]
    q = Quaternion(q[3], q[0], q[1], q[2])
    R = q.rotation_matrix
    
    if t < N-1:
        q2 = quats[t+1]
        q2 = Quaternion(q2[3], q2[0], q2[1], q2[2])
        wb.append(ComputeWb(q, q2))
    
    # velocity of center of mass
    b[t*3:t*3+3] = xyz
    A[t*3:t*3+3, 0:3] = I
    A[t*3:t*3+3, 3:6] = R
    A[t*3:t*3+3, 6:] = I * t

X = np.linalg.pinv(A) @ b
q_cm = X[:3]
q_b = X[3:6]
v_cm = X[6:] 
print(f"\na) v_cm: {v_cm}")


# q3.2 - inertia tensor
A = np.zeros(shape=(3*N-6, 6), dtype=np.float)
wb = np.array(wb)

for t in range(N-2):
    Rt = Rotation.from_rotvec(wb[t]).as_matrix()
    qt = Quaternion(matrix=Rt)
    
    Rtp1 = Rotation.from_rotvec(wb[t+1]).as_matrix()
    qtp1 = Quaternion(matrix=Rtp1)
    
    qd = qtp1 * qt.inverse
    
    theta = 2 * math.acos(qd.real)
    wb_dot = np.zeros(shape=3)
    if not np.isclose(theta, 0):
        wb_dot = qd.imaginary / math.sin(theta/2)
        wb_dot = theta * wb_dot / np.linalg.norm(wb_dot)
    
    w_dot_tilde = Tilde(wb_dot)
    w_hat = Skew(wb[t])
    w_tilde = Tilde(wb[t])

    A[t*3:t*3+3] = w_dot_tilde + w_hat @ w_tilde

u, s, vt = np.linalg.svd(A.T @ A)
I = vt[-1] / vt[-1, -1]
I_tensor = np.array([
    [I[0], I[1], I[2]],
    [I[1], I[3], I[4]],
    [I[2], I[4], I[5]]])
print(f"\nb) inertia tensor:\n{I_tensor}")

# q3.3 - transforming inertia tensor to 
I = np.linalg.eigvals(I_tensor)
I_tensor_p = I * np.identity(3)
print(f"\nc)principal axis inertia tensor:\n{I_tensor_p}")

# q3.4 - distance from center of mass and beacon
print(f"\nd) offset q_cm to q_b: {q_b} norm: {np.linalg.norm(q_b)}")

# q3.5 - angular momentum
# H = R I_b w_b
H = np.zeros(shape=3)
for t in range(len(wb)):
    q = quats[t]
    q = Quaternion(q[3], q[0], q[1], q[2])
    R = q.rotation_matrix
    H += R @ I_tensor @ wb[t]
print(f"\ne) angular momentum: {H / len(wb)}")

# q3.6 - asteroid pose after 120s
T_120 = T.tolist()
points_120 = points.tolist()
quats_120 = quats.tolist()

# measure the error using this method
err = np.zeros(shape=3)
for t in range(2, 61):
    
    qtm1 = quats_120[t-1]    
    qtm1 = Quaternion(qtm1[3], qtm1[0], qtm1[1], qtm1[2])

    qtm2 = quats_120[t-2]
    qtm2 = Quaternion(qtm2[3], qtm2[0], qtm2[1], qtm2[2])
    
    qd = qtm1 * qtm2.inverse
    theta = 2 * math.acos(qd.real)
    wtm1 = np.zeros(shape=3)
    if not np.isclose(theta, 0):
        wtm1 = qd.imaginary / math.sin(theta/2)
        wtm1 = theta * wtm1 / np.linalg.norm(wtm1)
    wtm1 = Quaternion(0, wtm1[0], wtm1[1], wtm1[2])
    
    # numerical
    qt = qtm1 + 0.5 * wtm1 * qtm1
    qt = qt.normalised
    
    # measured 
    qt_m = quats[t]
    qt_m = Quaternion(qt_m[3], qt_m[0], qt_m[1], qt_m[2])
    print(f"reconstructed\tqt: {qt}")
    print(f"measured\tqt: {qt_m}")
    err += (qt_m * qt.inverse).imaginary

err = err / 58
print(f"\terror: {err}\n\tnorm: {np.linalg.norm(err)}")
    
for t in range(61, 121):
    T_120.append(t)
    
    qtm1 = quats_120[t-1]
    
    qtm1 = Quaternion(qtm1[3], qtm1[0], qtm1[1], qtm1[2])
    
    qtm2 = quats_120[t-2]
    qtm2 = Quaternion(qtm2[3], qtm2[0], qtm2[1], qtm2[2])
    
    qd = qtm1 * qtm2.inverse
    theta = 2 * math.acos(qd.real)
    wtm1 = np.zeros(shape=3)
    if not np.isclose(theta, 0):
        wtm1 = qd.imaginary / math.sin(theta/2)
        wtm1 = theta * wtm1 / np.linalg.norm(wtm1)
    wtm1 = Quaternion(0, wtm1[0], wtm1[1], wtm1[2])
    
    # numerical
    qt = qtm1 + 0.5 * wtm1 * qtm1
    qt = qt.normalised
    Rt = qt.rotation_matrix
    
    qt = qt.elements
    quats_120.append([qt[1], qt[2], qt[3], qt[0]])
    
    xyz = q_cm + Rt @ q_b + v_cm * t 
    points_120.append(xyz)

print(f"\nf) last pose t: {T_120[-1]} xyz: {points_120[-1]} quat: {quats_120[-1]}")

points_120 = np.asarray(points_120)  
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_120[:, 0], points_120[:, 1], points_120[:, 2])
plt.show()
plt.close()
