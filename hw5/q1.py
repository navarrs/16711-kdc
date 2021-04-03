
# -----------------------------------------------------------------------------#
# @author ingridn
# @brief  16-711 KDC Hw5 
# -----------------------------------------------------------------------------#

from sympy import *
init_printing(use_unicode=True)

# Symbols
l0 = Symbol("l0")
l1 = Symbol("l1")
l2 = Symbol("l2")
r0 = Symbol("r0")
r1 = Symbol("r1")
r2 = Symbol("r2")
theta1 = Symbol("theta1")
theta2 = Symbol("theta2")
theta3 = Symbol("theta3")
m1 = Symbol("m1")
m2 = Symbol("m2")
m3 = Symbol("m3")
Ix1 = Symbol("Ix1")
Ix2 = Symbol("Ix2")
Ix3 = Symbol("Ix3")
Iy1 = Symbol("Iy1")
Iy2 = Symbol("Iy2")
Iy3 = Symbol("Iy3")
Iz1 = Symbol("Iz1")
Iz2 = Symbol("Iz2")
Iz3 = Symbol("Iz3")

# ------------------------------------------------------------------------------
# Twists 
xi_1 = Matrix([0, 0, 0, 0, 0, 1])
xi_2 = Matrix([0, -l0, 0, -1, 0, 0])
xi_3 = Matrix([0, -l0, l1, -1, 0, 0])
print("Twist 1:")
pprint(xi_1.T)
print("Twist 2:")
pprint(xi_2.T)
print("Twist 3:")
pprint(xi_3.T)

# ------------------------------------------------------------------------------
# Homogeneous transformations
G_sl1 = G_sl2 = G_sl3 = eye(4)
G_sl1[2, -1] = r0
print("G_sl1")
pprint(G_sl1)

G_sl2[1, -1] = r1
G_sl2[2, -1] = l0
print("G_sl2")
pprint(G_sl2)

G_sl3[1, -1] = l1 + r2
G_sl3[2, -1] = l0
print("G_sl3")
pprint(G_sl3)

# ------------------------------------------------------------------------------
# Jacobians 
# @TODO: compute?
Jb_sl1 = Matrix([[0, 0, 0], 
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0],
                 [1, 0, 0]])
print("Jb_sl1")
pprint(Jb_sl1)
Jb_sl2 = Matrix([[-r1 * cos(theta2), 0, 0], 
                 [0, 0, 0],
                 [0, -r1, 0],
                 [0, -1, 0],
                 [-sin(theta2), 0, 0],
                 [cos(theta2), 0, 0]])
print("Jb_sl2")
pprint(Jb_sl2)
Jb_sl3 = Matrix([[-l2*cos(theta2) - r2*cos(theta2)*cos(theta3), 0, 0], 
                 [0, l1*sin(theta3), 0],
                 [0, -r2-l1*cos(theta3), -r2],
                 [0, -1, -1],
                 [-sin(theta2)*sin(theta3), 0, 0],
                 [cos(theta2)*cos(theta3), 0, 0]])
print("Jb_sl3")
pprint(Jb_sl3)

# ------------------------------------------------------------------------------
# Mass matrices 
M1 = zeros(6)
M1[0, 0] = M1[1, 1] = M1[2, 2] = m1
M1[3, 3] = Ix1
M1[4, 4] = Iy1
M1[5, 5] = Iz1
print("M1")
pprint(M1)

M2 = zeros(6)
M2[0, 0] = M2[1, 1] = M2[2, 2] = m2
M2[3, 3] = Ix2
M2[4, 4] = Iy2
M2[5, 5] = Iz2
print("M2")
pprint(M2)

M3 = zeros(6)
M3[0, 0] = M3[1, 1] = M3[2, 2] = m3
M3[3, 3] = Ix3
M3[4, 4] = Iy3
M3[5, 5] = Iz3
print("M3")
pprint(M3)

# ------------------------------------------------------------------------------
# Inertia Matrix 
M = Jb_sl1.T * M1 * Jb_sl1 + Jb_sl2.T * M2 * Jb_sl2 + Jb_sl3.T * M3 * Jb_sl3 
print("Inertia Matrix M")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        print(f"M_{i+1}{j+1}: {M[i, j]}")