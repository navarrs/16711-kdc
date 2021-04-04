# -----------------------------------------------------------------------------#
# @author ingridn
# @brief 16-711 KDC Hw3 - utility functions
# -----------------------------------------------------------------------------#
import numpy as np
from sympy import * 

def Skew(v):
    """
    Skew symmetric matrix for a vector in R^{3x1}:
        skew = | 0 -z  y |
               | z  0 -x |
               |-y  x  0 |
    """
    return Matrix([[0,    -v[2],  v[1]],
                   [v[2],     0, -v[0]],
                   [-v[1], v[0],    0]])

def Rodrigues(w_hat, theta):
    """
    Rodrigues:
        e^{w_hat theta} = I + w_hat sin(theta) + w_hat^{2} (1 - cos(theta))
    """
    return (
        eye(3)
        + w_hat * sin(theta)
        + w_hat * w_hat * (1 - cos(theta))
    )

def Twist(q, w):
    """
    Twist:
        xi = | w_hat   v | with v = -w x q
             |    0    0 | 
    """
    twist = zeros(4)
    twist[:3, :3] = Skew(w)
    twist[:3, -1] = -w.cross(q)
    return twist

def ExpTwist(xi, theta):
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
    I = eye(3)
    exp_xi = eye(4)
    
    v = Matrix([xi[0], xi[1], xi[2]])
    w = Matrix([xi[3], xi[4], xi[5]])
    w_hat = Skew(w)
    
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
        what_v = w_hat * v
        t = (I - R) * what_v + (w * w.T) * v * theta
        exp_xi[:3, -1] = t
        
    return exp_xi

def AdjInv(g):
    """
    Adjoint Transformation inverse:
        Ad = | R^T   -R^T p_hat |
             | 0      R^T       |
    """
    R = g[:3, :3]
    R_T = R.T

    p = g[:3, -1]
    p_hat = Skew(p)
   
    adj_inv = zeros(6)
    adj_inv[:3, :3] = R_T
    adj_inv[:3, 3:] = -R_T * p_hat
    adj_inv[3:, 3:] = R_T
    return simplify(adj_inv)


def FK(twists, thetas, g_st_0):
    """
    Forward Kinematics map:
        g_st_theta = e^{xi_1 theta_1} ... e^{xi_n theta_n} g_st_0
    """
    assert len(twists) == len(thetas), \
        f"err: twists size {len(twists)} != thetas size {len(thetas)}"

    g_st_theta = eye(4)
    for i in range(len(twists)):
        g_st_theta = g_st_theta * ExpTwist(twists[i], thetas[i])

    return simplify(g_st_theta @ g_st_0)

def ComputeMassMatrix(m, Ix, Iy, Iz):
    """
    Computes a mass matrix in the form 
    M = | m 0 0  0  0  0 |
        | 0 m 0  0  0  0 |
        | 0 0 m  0  0  0 |
        | 0 0 0 Ix  0  0 |
        | 0 0 0  0 Iy  0 |
        | 0 0 0  0  0 Iz |
    """
    M = zeros(6)
    M[0, 0] = M[1, 1] = M[2, 2] = m    
    M[3, 3] = Ix
    M[4, 4] = Iy
    M[5, 5] = Iz
    return M   

def ComputeInertiaMatrix(M_list, J_list, simp=True):
    """
    Computes the inertia matrix as:
        M = J1.T M1 J1 + ... + Jn.T Mn Jn
    note: assumes lists are in order.
    """
    M = zeros(3)
    assert len(M_list) == len(J_list), \
        f"Mass list {len(M_list)} and Jacobian list {len(J_list)} size mismatch"
        
    for i in range(len(M_list)):
        M += J_list[i].T * M_list[i] * J_list[i]
    
    if simp:
        return simplify(M)
    return M


def ComputeCoriollisMatrix(M, thetas, dthetas, simp=True):
    """
    Computes the coriollis matrix as:
    C = 0.5 sum_k=1^n (dpMij/dpthetak + dpMik/dpthetaj - dpMkj/dpthetai) dthetak
    """
    C = zeros(3)
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            for k in range(3):
                # compute the centrifugal forces using inertia matrix
                if thetas[k] == 0.0:
                    dMij = 0.0
                else:
                    dMij = diff(M[i, j], thetas[k])
                
                if thetas[j] == 0.0:
                    dMik = 0.0
                else:
                    dMik = diff(M[i, k], thetas[j])
                
                if thetas[i] == 0.0:
                    dMkj = 0.0
                else:
                    dMkj = diff(M[k, j], thetas[i])
                Fijk = (dMij + dMik - dMkj) / 2
                # print(f"F_{i+1}{j+1}{k+1}:")
                # pprint(F_ijk)
                C[i, j] += Fijk * dthetas[k]
    if simp:
        return simplify(C)
    return C

def ComputeExternalForces(m, h, g, thetas, simp=True):
    """
    Computes external forces vector
    """
    assert len(m) == len(h) == len(thetas), \
        f"Mass {len(m)}, height {len(h)}, thetas {len(thetas)} size mismatch"
    
    # compute the potential energy 
    V = 0
    n = len(m)
    for i in range(n):
        V += m[i] * h[i] * g
    
    # compute the effect of the gravitational forces
    N = zeros(n, 1)
    for i in range(n):
        if thetas[i] == 0.0:
            N[i] = 0.0
        else:
            N[i] = diff(V, thetas[i])
    
    if simp:
        return simplify(N)
    return N