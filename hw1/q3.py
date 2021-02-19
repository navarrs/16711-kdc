import matplotlib.pyplot as plt
import numpy as np 

from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import matrix_power

np.set_printoptions(4, suppress=True)

# Transformations between frames
# G base wrt world
G_base_world = np.array(
    [[1.0, 0.0, 0.0,  750.0],
     [0.0, 1.0, 0.0,  500.0],
     [0.0, 0.0, 1.0, 1000.0],
     [0.0, 0.0, 0.0,    1.0]], dtype=np.float)

# G shoulder wrt base
G_shl_base = np.array(
    [[1.0, 0.0, 0.0, 220.0],
     [0.0, 1.0, 0.0, 140.0],
     [0.0, 0.0, 1.0, 346.0],
     [0.0, 0.0, 0.0,   1.0]], dtype=np.float)

G_shl_world = G_shl_base @ G_base_world

# G end-effector wrt shoulder
G_eef_shl = np.array(
    [[1.0, 0.0, 0.0,   0.0],
     [0.0, 1.0, 0.0,   0.0],
     [0.0, 0.0, 1.0, 910.0],
     [0.0, 0.0, 0.0,   1.0]], dtype=np.float)

# G tool wrt end-effector
G_tool_eef = np.array(
    [[1.0, 0.0, 0.0,  0.0],
     [0.0, 1.0, 0.0,  0.0],
     [0.0, 0.0, 1.0, 120.0],
     [0.0, 0.0, 0.0,  1.0]], dtype=np.float)

# print(G_world_base @ G_base_shl @ G_shl_end @ G_end_tool)

def compute_twists(qw_matrix):
    twists = np.zeros((len(qw_matrix), 6), dtype=np.float)
    for i, qw in enumerate(qw_matrix):
        # sanity check
        q = qw[:3]
        w = qw[3:]
        twists[i, :3] = -np.cross(w, q)
        twists[i, 3:] = w
        # print(f"twist: {twists[i]} q: {q} w: {w}")  
    return twists

def skew(w):
    return np.array([[   0,  w[2], -w[1]],
                     [-w[2],     0, w[0]],
                     [ w[1], -w[0],   0]], dtype=np.float)

def compute_matrix_exponential_w(w_hat, theta):
    I = np.identity(3)
    # return I + w_hat * np.sin(theta) + w_hat @ w_hat * (1 - np.cos(theta))   
    return I + w_hat * np.sin(theta) + matrix_power(w_hat, 2) * (1 - np.cos(theta))    

def compute_matrix_exponential_xi(twist, theta):
    I = np.identity(3)
    exp_xi = np.identity(4)
    
    # w, q
    v = twist[:3]
    w = twist[3:]
    
    # compute the rotation part 
    w_hat = skew(w)
    R = compute_matrix_exponential_w(w_hat, theta)
    exp_xi[:3, :3] = R
    
    # compute the translation part
    # (I - exp_w) (wxv) + wwtvtheta = (I - exp_w) wxv
    w_x_v = np.cross(w, v)
    t = (I - R) @ w_hat @ v
    exp_xi[:3, -1] = t
    return exp_xi

qw = np.loadtxt("qw.txt", delimiter=',', dtype=np.float)
joint_data = np.loadtxt("JointData.txt", delimiter=' ', dtype=np.float)
twists = compute_twists(qw)
n_twists = len(twists)

# G = np.identity(4)
# for n in range(n_twists):
#     exp_xi = compute_matrix_exponential_xi(twists[n], 0)
#     G = G @  exp_xi
# G_eef_shl = G @ G_eef_shl
# print(G_tool_eef @ G_eef_shl @ G_shl_base @ G_base_world)

coords = np.zeros((len(joint_data), 3), dtype=np.float)
for j, joints in enumerate(joint_data):
    
    # find G_eef wrt shouler
    G_eef_shl_ = np.identity(4)
    for i, joint_angle in enumerate(joints):
        xi_exp = compute_matrix_exponential_xi(twists[i], joint_angle)
        G_eef_shl_ = G_eef_shl_ @ xi_exp
        
    # G_tool_world
    G_tool_world = G_tool_eef @ G_eef_shl_ @ G_shl_world
    
    coords[j, 0] = G_tool_world[0, 3] # x
    coords[j, 1] = G_tool_world[1, 3] # y 
    coords[j, 2] = G_tool_world[2, 3] # z
print(coords.shape)
#     print(x[-1], y[-1], z[-1])
    
np.savetxt("marker_coords.txt", coords, fmt="%10.5f", delimiter=' ')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(coords[:, 0], coords[:, 1], coords[:, 2])
plt.savefig("marker_drawing.png")
plt.show()