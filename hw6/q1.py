# ------------------------------------------------------------------------------
# @author  ingridn
# @brief   Hw6
# ------------------------------------------------------------------------------
from sympy import * 
from sympy.solvers import solve
from scipy import signal
from scipy.integrate import solve_ivp

import numpy as np
import matplotlib.pyplot as plt 
# ------------------------------------------------------------------------------
# Symbols

# q1.a
print("\n#", "-"*50)
print("q1 a) state space model")
J1, J2, ki, k, c, s = symbols("J1,J2,ki,k,c,s")
x1, x2, x3, x4 = symbols("x1,x2,x3,x4")
k1, k2, k3, k4 = symbols("k1,k2,k3,k4")

w0 = sqrt(k * (J1 + J2) / (J1 * J2))
A = Matrix([[  0.0,   0.0,       w0,      0.0],
            [  0.0,   0.0,      0.0,       w0],
            [-k/J1,  k/J1, -w0*c/J1,  w0*c/J1],
            [ k/J2, -k/J2,  w0*c/J2, -w0*c/J2]])
X = Matrix([x1, x2, x3, x4])
B = Matrix([0, 0, 1/J1, 0])
B_dist = Matrix([0, 0, 0, 1/J2])         
K = Matrix([k1, k2, k3, k4]).T

print("A:")
pprint(A)

print("B:")
pprint(B)

print("B_dist:")
pprint(B_dist)

# q1.b
print("\n#", "-"*50)
print("q1 b) normalized parameters")
_J1 = 10.0 / 9.0
_J2 = 10.0
_c = 0.1
_k = 1.0
_ki = 1.0
w0 = w0.subs({J1: _J1, J2: _J2, c: _c, k: _k, ki: _ki})
print("params:\n")
print(f"\tJ1: {_J1}\n\tJ2: {_J2}\n\tc: {_c}\n\tJ1: {_J1}\n\tk: {_k}\n\tki: {_ki}\n\tw0: {w0}")

A = A.subs({J1: _J1, J2: _J2, c: _c, k: _k, ki: _ki, w0: w0})

print("A:")
pprint(A)

print("eigen values of A:")
pprint(A.eigenvals())

# q1.c
print("\n#", "-"*50)
print("q1 c) state feedback loop")

A = np.asarray(A).astype(np.float64)
print(f"A:\n{A}")

B = np.asarray(B.subs({J1: _J1})).astype(np.float64)
print(f"B:\n{B}")

poles = [-2, -1, -1+1j, -1-1j]
K = signal.place_poles(A, B, poles, method='YT').gain_matrix.T
print(f"K:\n{K}")

# q1 d)
print("\n#", "-"*50)
print("q1 d) simulate state feedback loop")

niter = 50
x_ref = np.zeros(shape=(4, 1))
x = np.zeros(shape=(4, 1))

td = np.array([0, 0, 1, 0], dtype=np.float64).reshape(4, 1)
B_dist = np.asarray(B_dist.subs({J2: _J2})).astype(np.float64).reshape(4, 1)

Bd = B_dist 

C = np.array([1, 1, 0, 0]).reshape(4, 1)

def Int(t, x):
    x1, x2, dx1, dx2 = x
    return x1, x2, dx1, dx2

t0, dt = 0, 0.05
T = [t0]
Y = []
for i in range(niter):
    
    t = [t0, t0 + dt]
    t0 += dt
    T.append(t0)
    
    x_e = np.multiply(K, x)
    u = x_ref - x_e
    
    x_dot = (np.dot(A, x) + B * u +  Bd).reshape(4)
    
    # print("ax: ", np.dot(A, x))
    # print("bu: ", B * u)
    # print("Bd: ", Bd)
    
    sol = solve_ivp(Int, t, x_dot)
    x[0], x[1] = sol.y[0, -1], sol.y[1, -1]
    x[2], x[3] = sol.y[2, -1], sol.y[3, -1]
    
    y = C * x
    Y.append([y[0, 0], y[1, 0]])

Y = np.asarray(Y)

# plt.plot(Y[:, 0], label='x1')
plt.plot(Y[:, 1], label='x2')
plt.xlabel('t')
plt.legend()
plt.title('Step response')
plt.show()