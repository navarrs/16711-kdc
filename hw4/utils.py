# -----------------------------------------------------------------------------#
# @author ingridn
# @brief 16-711 KDC Hw4 - utility functions
# -----------------------------------------------------------------------------#
import numpy as np

def Tilde(w):
    """
        Expanded tilde matrix used to get the inerta tensor
    """
    return np.array([
        [w[0], w[1], w[2], 0, 0, 0],
        [0, w[0], 0, w[1], w[2], 0],
        [0, 0, w[0], 0, w[1], w[2]]], dtype=np.float)
    
def Skew(v):
    """
    Skew symmetric matrix for a vector in R^{3x1}:
        skew = | 0 -z  y |
               | z  0 -x |
               |-y  x  0 |
    """
    return np.array([[0,   -v[2], v[1]],
                     [v[2],    0, -v[0]],
                     [-v[1], v[0],   0]], dtype=np.float)

def Skew2Vec(M):
    """
    Skew matrix to vector:
        x = M[2, 1]
        y = M[0, 2]
        z = M[1, 0]
    """
    return np.array([M[2, 1], M[0, 2], M[1, 0]])

def Rodrigues(w_hat, theta):
    """
    Rodrigues:
        e^{w_hat theta} = I + w_hat sin(theta) + w_hat^{2} (1 - cos(theta))
    """
    return (
        np.identity(3)
        + w_hat * np.sin(theta)
        + np.linalg.matrix_power(w_hat, 2) * (1 - np.cos(theta))
    )

def Twist(q, w):
    """
    Twist:
        xi = | w_hat   v | with v = -w x q
             |    0    0 | 
    """
    twist = np.zeros(shape=(4, 4), dtype=np.float)
    twist[:3, :3] = Skew(w)
    twist[:3, -1] = -np.cross(w, q)
    return twist
    

def Twist2Vec(twist):
    """
    Twist to vector:
        xi = | -w x q | 
             |    w   | 
    """
    xi_vec = np.zeros(shape=(6), dtype=np.float)
    xi_vec[:3] = twist[:3, -1]
    xi_vec[3:] = Skew2Vec(twist[:3, :3])
    return xi_vec


def ExpTwist(twist, theta):
    """
    Exponential of a twist:
        case rotation and translation:
            R = rogrigues(w_hat, theta)
            exp^{xi theta} = 
                | R   (I - R)(w X v) + w w^{T} v theta |
                | 0                  1                 |      
        case pure translation      
            exp^{xi theta} = 
                    | I   v theta |
                    | 0      1    |      
    """
    I = np.identity(3)
    exp_xi = np.identity(4)

    # Get components of twist
    w_hat = twist[:3, :3]
    w = Skew2Vec(w_hat)
    v = twist[:3, -1]

    if np.all(w == 0):
        exp_xi[:3, :3] = I
        exp_xi[:3, -1] = v * theta
    else:
        # compute the rotation part
        R = Rodrigues(w_hat, theta)
        exp_xi[:3, :3] = R

        # compute the translation part
        # (I - exp_w) (wxv) + wwtvtheta = (I - exp_w) wxv
        # w_x_v = w_hat v
        # w_x_v = np.cross(w, v)
        what_v = w_hat @ v
        t = (I - R) @ what_v + w * np.dot(w, v) * theta
        exp_xi[:3, -1] = t
    return exp_xi


def Adj(g):
    """
    Adjoint Transformation is:
        Ad = | R   p_hat R |
             | 0      R    |
    """
    R = g[:3, :3]
    p_hat = Skew(g[:3, -1])

    adj = np.zeros((6, 6), dtype=np.float)
    adj[:3, :3] = R
    adj[:3, 3:] = p_hat @ R
    adj[3:, 3:] = R
    return adj


def AdjInv(g):
    """
    Adjoint Transformation inverse:
        Ad = | R^T   -R^T p_hat |
             | 0      R^T       |
    """
    R_T = g[:3, :3].T
    p_hat = Skew(g[:3, -1])

    adj_inv = np.zeros((6, 6), dtype=np.float)

    adj_inv[:3, :3] = R_T
    adj_inv[:3, 3:] = -R_T @ p_hat
    adj_inv[3:, 3:] = R_T
    return adj


def FK(twists, thetas, g_st_0):
    """
    Forward Kinematics map:
        g_st_theta = e^{xi_1 theta_1} ... e^{xi_n theta_n} g_st_0
    """
    assert len(twists) == len(thetas), \
        f"err: twists size {len(twists)} != thetas size {len(thetas)}"

    g_st_theta = np.identity(n=4, dtype=np.float)
    for i in range(len(twists)):
        g_st_theta = g_st_theta @ ExpTwist(twists[i], thetas[i])

    return g_st_theta @ g_st_0

def Jacobian(twists, thetas):
    """ 
    Jacobian in spatial frame:
        J = [xi_1, x_2' ... xi_n']
        where xi_j = Adj(e^{xi_1 theta_1}...e^{xi_j-1 theta_j-1}) xi_j
    """
    assert len(twists) == len(thetas), \
        f"err: twists size {len(twists)} != thetas size {len(thetas)}"
    
    N = len(thetas)
    
    # Jacobian 
    J = np.zeros(shape=(6, N), dtype=np.float)
    
    # Compute exponentials of twists
    exp_xis = np.zeros(shape=(N, 4, 4), dtype=np.float)
    for i in range(N):
        exp_xis[i] = ExpTwist(twists[i], thetas[i])
        # print(f"Exp_xi {i}\n{exp_xis[i]}")
    
    # First column of J is the vector of twist 1
    J[:, 0] = Twist2Vec(twists[0])
    
    # The next columns are Adj[1...(j-1)] @ twist_j
    exp_temp = np.identity(n=4, dtype=np.float)
    adj = np.identity(6, dtype=np.float)
    for j in range(N):
        if j > 0:
            exp_temp = exp_temp @ exp_xis[j-1]
            adj = Adj(exp_temp)
        J[:, j] = adj @ Twist2Vec(twists[j])
        
    return J

def JacobianPInv(J):
    """
    Jacobian Pseudo Inverse:
        J_pinv = J^T (JJ^T)^{-1}
    """
    return J.T @ np.linalg.inv(J @ J.T)

def JacobianInv(J, lamb = 0.0):
    """
    Jacobian damped inverse
        J_inv = J^T (J @ J^T + lambda^2 I)
        if lamb = 0 then its the same as before
    """
    I = np.identity(len(J))
    return J.T @ np.linalg.inv(J @ J.T + lamb ** 2 * I)