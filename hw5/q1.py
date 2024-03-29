
# -----------------------------------------------------------------------------#
# @author ingridn
# @brief  16-711 KDC Hw5 
# -----------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import numpy as np

from math import pi
from utils import (
    AdjInv, 
    ComputeCoriollisMatrix,
    ComputeExternalForces,
    ComputeInertiaMatrix,
    ComputeMassMatrix,
    ExpTwist, 
    FK,
    Twist, 
)
from scipy.integrate import solve_ivp
from sympy import *
init_printing(use_unicode=True)

forward_dynamics = True

# ------------------------------------------------------------------------------
# Symbols
tau,g,t = symbols("tau,g,t")
l0, l1, l2 = symbols("l0,l1,l2")
r0, r1, r2 = symbols("r0,r1,r2")
theta1, theta2, theta3 = symbols("theta1,theta2,theta3")
dtheta1, dtheta2, dtheta3 = symbols("dtheta1,dtheta2,dtheta3")
m1, m2, m3 = symbols("m1,m2,m3")
Ix1,Iy1,Iz1 = symbols("Ix1,Iy1,Iz1")
Ix2,Iy2,Iz2 = symbols("Ix2,Iy2,Iz2")
Ix3,Iy3,Iz3 = symbols("Ix3,Iy3,Iz3")

# ------------------------------------------------------------------------------
# Config
if forward_dynamics:
    C = {
        "l0": 1, "l1": 1, "l2": 1,
        "r0": 0.5, "r1": 0.5, "r2": 0.5,
        "m1": 1, "m2": 1, "m3": 1,
        "Ix1": 1, "Iy1": 1, "Iz1": 1,
        "Ix2": 1, "Iy2": 1, "Iz2": 1,
        "Ix3": 1, "Iy3": 1, "Iz3": 1,
        "g": 9.81
    }
    l0, l1, l2 = C["l0"], C["l1"], C["l2"]
    r0, r1, r2 = C["r0"], C["r1"], C["r2"]
    m1, m2, m3 = C["m1"], C["m2"], C["m3"] 
    Ix1, Iy1, Iz1 = C["Ix1"], C["Iy1"], C["Iz1"]
    Ix2, Iy2, Iz2 = C["Ix2"], C["Iy2"], C["Iz2"]
    Ix3, Iy3, Iz3 = C["Ix3"], C["Iy3"], C["Iz3"]
    g = C["g"]

# ------------------------------------------------------------------------------
# Twists 
xi_1 = Matrix([0, 0, 0, 0, 0, 1])
xi_2 = Matrix([0, -l0, 0, -1, 0, 0])
xi_3 = Matrix([0, -l0, l1, -1, 0, 0])
print("Twist 1:")
pprint(xi_1)
print("Twist 2:")
pprint(xi_2)
print("Twist 3:")
pprint(xi_3)

# ------------------------------------------------------------------------------
# Homogeneous transformations
G0_sl1 = eye(4)
G0_sl1[2, -1] = r0
print("G0_sl1")
pprint(G0_sl1)

G0_sl2 = eye(4)
G0_sl2[1, -1] = r1
G0_sl2[2, -1] = l0
print("G0_sl2")
pprint(G0_sl2)

G0_sl3 = eye(4)
G0_sl3[1, -1] = l1 + r2
G0_sl3[2, -1] = l0
print("G0_sl3")
pprint(G0_sl3)

# ------------------------------------------------------------------------------
# Exponential twists 
exp_xi1 = simplify(ExpTwist(xi_1, theta1))
# print("exp_xi1")
# pprint(exp_xi1)

exp_xi2 = simplify(ExpTwist(xi_2, theta2))
# print("exp_xi2")
# pprint(exp_xi2)

exp_xi3 = simplify(ExpTwist(xi_3, theta3))
# print("exp_xi3")
# pprint(exp_xi3)

# ------------------------------------------------------------------------------
# Jacobians 
# body jacobian 1
J1 = Matrix([[0, 0, 0], 
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [1, 0, 0]])
# print("J1")
# pprint(J1)

# body jacobian 2
G1 = simplify(exp_xi1 * exp_xi2 * G0_sl2)
xi_dagger1 = AdjInv(G1) * xi_1

G2 = simplify(exp_xi2 * G0_sl2)
xi_dagger2 = AdjInv(G2) * xi_2

J2 = zeros(6, 3)
J2[:, 0] = xi_dagger1
J2[:, 1] = xi_dagger2
# print("\nJ2")
# pprint(J2)

# body jacobian 3
G1 = simplify(exp_xi1 * exp_xi2 * exp_xi3 * G0_sl3)
xi_dagger1 = AdjInv(G1) * xi_1

G2 = simplify(exp_xi2 * exp_xi3 * G0_sl3)
xi_dagger2 = AdjInv(G2) * xi_2

G3 = simplify(exp_xi3 * G0_sl3)
xi_dagger3 = AdjInv(G3) * xi_3

J3 = zeros(6, 3)
J3[:, 0] = xi_dagger1
J3[:, 1] = xi_dagger2
J3[:, 2] = xi_dagger3
# print("\nJ3")
# pprint(J3)

# ------------------------------------------------------------------------------
# Mass matrices 
M1 = ComputeMassMatrix(m1, Ix1, Iy1, Iz1)
# print("\nM1")
# pprint(M1)

M2 = ComputeMassMatrix(m2, Ix2, Iy2, Iz2)
# print("\nM2")
# pprint(M2)

M3 = ComputeMassMatrix(m3, Ix3, Iy3, Iz3)
# print("\nM3")
# pprint(M3)

# ------------------------------------------------------------------------------
# Inertia Matrix 
M_list = [M1, M2, M3]
J_list = [J1, J2, J3]
M = ComputeInertiaMatrix(M_list, J_list)

print(f"\n#", "-"*50)
print("\nInertia Matrix M\n")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        print(f"\nM_{i+1}{j+1}:")
        pprint(simplify(M[i, j]))
        
# ------------------------------------------------------------------------------
# Coriolis Matrix 
thetas = [theta1, theta2, theta3]
dthetas = [dtheta1, dtheta2, dtheta3]
C = ComputeCoriollisMatrix(M, thetas, dthetas, False)

print(f"\n#", "-"*50)
print("\nCoriollis Matrix C\n")
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        print(f"\nC_{i+1}{j+1}:")
        pprint(C[i, j])

# ------------------------------------------------------------------------------
# Gravitational 
print(f"\n#", "-"*50)
print(f"\nGravitational component N")

# get the corresponding heights by using forward kinematics and extracing the 
# z component from the resulting transformation
twists = [xi_1]
thetas = [theta1]
G_sl1 = FK(twists, thetas, G0_sl1)
h1 = G0_sl1[2, -1]
print("h1:")
pprint(h1)

twists.append(xi_2)
thetas.append(theta2)
G_sl2 = FK(twists, thetas, G0_sl2)
h2 = G_sl2[2, -1]
print("h2:")
pprint(h2)

twists.append(xi_3)
thetas.append(theta3)
G_sl3 = FK(twists, thetas, G0_sl3)
h3 = G_sl3[2, -1]
print("h3:")
pprint(h3)

m = [m1, m2, m3]
h = [h1, h2, h3]

N = ComputeExternalForces(m, h, g, thetas)
for i in range(len(m)):
    print(f"\nN_{i+1}:")
    pprint(N[i])


# ------------------------------------------------------------------------------
# Forward Dynamics
def IntAcc(t, X, ax):
    x, dx = X
    return dx, ax

if forward_dynamics:
    print(f"\n#", "-"*50)
    print(f"\nForward dynamics")

    # Computing initial configuration
    eps = 1e-3
    
    q_d = Matrix([0.0, 0.0, 0.0])
    q0 = Matrix([0.0, pi/2, 0.0])
    q = Matrix([0.0, pi/2, 0.0])
    
    dq0 = Matrix([0.0, -0.01, 0.0])
    dq = Matrix([0.0, -0.01, 0.0])
    ddq = Matrix([0.0, -0.01, 0.0])
    
    # subsitute with current values
    Minv = M.inv()
    Mtinv = Minv.subs({theta2 : q0[1], theta3: q0[2]})
    Mt = M.subs({theta2 : q0[1], theta3: q0[2]})
    Ct = C.subs({theta2 : q0[1], theta3: q0[2], dtheta2: dq[1], dtheta3: dq[2], 
                 dtheta1: dq[0]})
    Nt = N.subs({theta2 : q0[1], theta3: q0[2]})
    tau = Mt * ddq + Ct * dq + Nt 

    err = q - q_d
    e = sqrt(err.dot(err))
    max_iter = 1000
    i = 0
    t0, dt = 0, 0.05
    Q = [[q[0], q[1], q[2]]]
    T = [t0]
    while e > eps and max_iter > i:
        i += 1
        
        t = [t0, t0 + dt]
        t0 += dt
        T.append(t0)

        # compute acceleration components
        ddq = Mtinv * (tau - Ct * dq - Nt)
    
        # numerical integration
        sol = solve_ivp(IntAcc, t, [q[0], dq[0]], args=[ddq[0]])
        q[0], dq[0] = sol.y[0, -1], sol.y[1, -1]
        sol = solve_ivp(IntAcc, t, [q[1], dq[1]], args=[ddq[1]])
        q[1], dq[1] = sol.y[0, -1], sol.y[1, -1]
        sol = solve_ivp(IntAcc, t, [q[2], dq[2]], args=[ddq[2]])
        q[2], dq[2] = sol.y[0, -1], sol.y[1, -1]
        Q.append([q[0], q[1], q[2]])
        
        pprint(q)
        # dq = dq0 + ddq * t
        # q = q0 + dq0 * t + 0.5 * ddq * t ** 2
        
        err = q - q_d
        e = sqrt(err.dot(err))
        print(f"iter: {i} error: {e}")
        
        # substitute with current values
        Mtinv = Minv.subs({theta2 : q[1], theta3: q[2]})
        Mt = M.subs({theta2 :q[1], theta3: q[2]})
        Ct = C.subs({theta2: q[1], theta3: q[2], 
                    dtheta1: dq[0], dtheta2: dq[1], dtheta3: dq[2]})
        Nt = N.subs({theta2 : q[1], theta3: q[2]})
        tau = Mt * ddq + Ct * dq + Nt 
        
    # Plot
    Q = np.asarray(Q)
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle('Joint angles over time')
    axs[0].plot(T, Q[:, 0])
    axs[0].set_title('theta0')
    axs[0].set_ylabel('angle (rad)')
    axs[1].plot(T, Q[:, 1])
    axs[1].set_title('theta1')
    axs[1].set_ylabel('angle (rad)')
    axs[2].plot(T, Q[:, 2])
    axs[2].set_title('theta2')
    axs[2].set_ylabel('angle (rad)')
    axs[2].set_xlabel('time')
    plt.show()
    plt.close()