import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from numpy.linalg import matrix_power
from scipy import linalg

np.set_printoptions(4, suppress=True)

from utils import EstimatePlane, RANSAC

# ------------------------------------------------------------------------------
# Load data

# G base wrt world
G_world_base = np.loadtxt(
    "data/g_world_base.txt", delimiter=" ", dtype=np.float)
print(f"G base wrt world:\n{G_world_base}")

# G shoulder wrt base
G_base_shoulder = np.loadtxt(
    "data/g_base_shoulder.txt", delimiter=" ", dtype=np.float)
print(f"G shoulder wrt base:\n{G_base_shoulder}")

# G shoulder wrt world
G_world_shoulder = G_world_base @ G_base_shoulder
print(f"G shoulder wrt world:\n{G_world_shoulder}")

# G tool wrt shoulder
G_shoulder_tool = np.loadtxt(
    "data/g_shoulder_tool.txt", delimiter=" ", dtype=np.float)
print(f"G tool wrt shoulder:\n{G_shoulder_tool}")

# q and w vector to compute the twists
qw = np.loadtxt("data/qw.txt", delimiter=' ', dtype=np.float)
print(f"q and w vectors:\n{qw}")

joint_data = np.loadtxt("data/joint_data.txt", delimiter=' ', dtype=np.float)

# ------------------------------------------------------------------------------
# Methods

def compute_twists(qw_matrix):
    twists = np.zeros((len(qw_matrix), 6), dtype=np.float)
    for i, qw in enumerate(qw_matrix):
        # sanity check
        q = qw[:3]
        w = qw[3:]
        twists[i, :3] = -np.cross(w, q)
        twists[i, 3:] = w
    return twists


def skew(w):
    return np.array([[0,  -w[2], w[1]],
                     [w[2],  0, -w[0]],
                     [-w[1], w[0], 0]], dtype=np.float)


def rodrigues(w_hat, theta):
    return (
        np.identity(3) 
        + w_hat * np.sin(theta) 
        + matrix_power(w_hat, 2) * (1 - np.cos(theta))
    )


def compute_matrix_exponential_xi(twist, w_hat, theta):
    I = np.identity(3)
    exp_xi = np.identity(4)

    # w, q
    v = twist[:3]
    w = twist[3:]

    # compute the rotation part
    R = rodrigues(w_hat, theta)
    exp_xi[:3, :3] = R

    # compute the translation part
    # (I - exp_w) (wxv) + wwtvtheta = (I - exp_w) wxv
    w_x_v = np.cross(w, v)
    t = (I - R) @ w_x_v + w * np.dot(w, v) * theta
    exp_xi[:3, -1] = t
    return exp_xi

# ------------------------------------------------------------------------------
# Main Program

def main():
    twists = compute_twists(qw)
    n_twists = len(twists)
    
    w_hats = np.zeros(shape=(n_twists, 3, 3), dtype=np.float)
    for n in range(n_twists):
        w_hats[n] = skew(twists[n, 3:])
    
    print(f"Twists:\n{twists}")

    coords = np.zeros((len(joint_data), 3), dtype=np.float)
    for j, joints in enumerate(joint_data):

        # Compute matrix exponentials 
        xi_exp = np.identity(4)
        for i, joint_angle in enumerate(joints):
            xi_exp = xi_exp @ compute_matrix_exponential_xi(
                twists[i], w_hats[i], joint_angle)
            
        # Compute G tool wrt shoulder 
        G_shoulder_tool_ = xi_exp @ G_shoulder_tool
        
        # Compute G tool wrt world 
        G_world_tool = G_world_shoulder @ G_shoulder_tool_

        # Get tool coordinate 
        coords[j] = G_world_tool[:3, -1]
    
    n, _, _ = RANSAC(coords)
    print(f"Normal: {n}")

    # Save tool trajectory
    np.savetxt("data/marker_coords.txt", coords, fmt="%10.5f", delimiter=' ')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot3D(coords[:, 0], coords[:, 1], coords[:, 2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("robot_drawing")
    plt.savefig("data/marker_drawing.png")
    plt.show()


if __name__ == "__main__":
    main()
