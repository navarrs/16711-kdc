
# -----------------------------------------------------------------------------#
# @author ingridn
# @brief  16-711 KDC Hw5 
# -----------------------------------------------------------------------------#

from utils import ExpTwist, Twist, AdjInv, FK
from sympy import *
init_printing(use_unicode=True)

# Symbols
t,g = symbols("t,g")
l0, l1, l2 = symbols("l0,l1,l2")
r0, r1, r2 = symbols("r0,r1,r2")
theta1, theta2, theta3 = symbols("theta1,theta2,theta3")
dtheta1, dtheta2, dtheta3 = symbols("dtheta1,dtheta2,dtheta3")
m1, m2, m3 = symbols("m1,m2,m3")
Ix1,Iy1,Iz1 = symbols("Ix1,Iy1,Iz1")
Ix2,Iy2,Iz2 = symbols("Ix2,Iy2,Iz2")
Ix3,Iy3,Iz3 = symbols("Ix3,Iy3,Iz3")

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
print("exp_xi1")
pprint(exp_xi1)

exp_xi2 = simplify(ExpTwist(xi_2, theta2))
print("exp_xi2")
pprint(exp_xi2)

exp_xi3 = simplify(ExpTwist(xi_3, theta3))
print("exp_xi3")
pprint(exp_xi3)

# # ------------------------------------------------------------------------------
# Jacobians 
# body jacobian for 1
J1 = Matrix([[0, 0, 0], 
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [1, 0, 0]])
print("J1")
pprint(J1)

# body jacobian for 2

G1 = simplify(exp_xi1 * exp_xi2 * G0_sl2)
xi_dagger1 = AdjInv(G1) * xi_1

G2 = simplify(exp_xi2 * G0_sl2)
xi_dagger2 = AdjInv(G2) * xi_2

J2 = zeros(6, 3)
J2[:, 0] = xi_dagger1
J2[:, 1] = xi_dagger2

print("\nJ2")
pprint(J2)

# body jacobian for 3
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

print("\nJ3")
pprint(J3)

# ------------------------------------------------------------------------------
# Mass matrices 
M1 = zeros(6)
M1[0, 0] = M1[1, 1] = M1[2, 2] = m1
M1[3, 3] = Ix1
M1[4, 4] = Iy1
M1[5, 5] = Iz1
print("\nM1")
pprint(M1)

M2 = zeros(6)
M2[0, 0] = M2[1, 1] = M2[2, 2] = m2
M2[3, 3] = Ix2
M2[4, 4] = Iy2
M2[5, 5] = Iz2
print("\nM2")
pprint(M2)

M3 = zeros(6)
M3[0, 0] = M3[1, 1] = M3[2, 2] = m3
M3[3, 3] = Ix3
M3[4, 4] = Iy3
M3[5, 5] = Iz3
print("\nM3")
pprint(M3)

# # ------------------------------------------------------------------------------
# # Inertia Matrix 
M = J1.T * M1 * J1 + J2.T * M2 * J2 + J3.T * M3 * J3 
print("\nInertia Matrix M\n")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        print(f"M_{i+1}{j+1}:")
        pprint(simplify(M[i, j]))
        
# ------------------------------------------------------------------------------
# Coriolis Matrix 
C = zeros(3)
thetas = [theta1, theta2, theta3]
dthetas = [dtheta1, dtheta2, dtheta3]
print("\nCoriollis Matrix C\n")
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        for k in range(3):
            dM_ij = diff(M[i, j], thetas[k])
            dM_ik = diff(M[i, k], thetas[j])
            dM_kj = diff(M[k, j], thetas[i])
            F_ijk = (dM_ij + dM_ik - dM_kj)/2
            print(f"F_{i+1}{j+1}{k+1}:")
            pprint(F_ijk)
            C[i, j] += F_ijk * dthetas[k]
        # print(f"C_{i+1}{j+1}: {C[i, j]}")

# ------------------------------------------------------------------------------
# Gravitational 
print(f"\nGravitaional component N")

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

V = m1*g*h1 + m2*g*h2 + m3*g*h3
N = Matrix([0, 0, 0])
N[0] = simplify(diff(V, theta1))
print("\nn1")
pprint(N[0])
N[1] = simplify(diff(V, theta2))
print("\nn2")
pprint(N[1])
N[2] = simplify(diff(V, theta3))
print("\nn3")
pprint(N[2])