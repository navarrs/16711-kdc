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
np.set_printoptions(suppress=True)

# ------------------------------------------------------------------------------
# q1.a
print("\n#", "-"*50)
print("q1 a) state space model")
J1, J2, ki, k, c, s = symbols("J1,J2,ki,k,c,s")
x1, x2, x3, x4 = symbols("x1,x2,x3,x4")

w0 = sqrt(k * (J1 + J2) / (J1 * J2))
A = Matrix([[0.0,   0.0,       w0,      0.0],
            [0.0,   0.0,      0.0,       w0],
            [-k/J1,  k/J1, -w0*c/J1,  w0*c/J1],
            [k/J2, -k/J2,  w0*c/J2, -w0*c/J2]])
X = Matrix([x1, x2, x3, x4])
B = Matrix([0, 0, ki/J1, 0])

# distortion
B_dist = Matrix([0, 0, 0, 1/J2])

print("A:")
pprint(A)

print("B:")
pprint(B)

print("B_dist:")
pprint(B_dist)

# ------------------------------------------------------------------------------
# q1.b
print("\n#", "-"*50)
print("q1 b) parameters")

_J1 = 10.0 / 9.0
_J2 = 10.0
_c = 0.1
_k = 1.0
_ki = 1.0
w0 = w0.subs({J1: _J1, J2: _J2, c: _c, k: _k, ki: _ki})
print("params:\n")
print(f"\tJ1: {_J1}\n\tJ2: {_J2}\n\tc: {_c}\n\tk: {_k}\n\tki: {_ki}\n\tw0: {w0}")

A = A.subs({J1: _J1, J2: _J2, c: _c, k: _k, ki: _ki, w0: w0})

print("A:")
pprint(A)

poles_open_loop = np.array([k for k in A.eigenvals().keys()], dtype=complex)
print(f"open-loop poles:\n{poles_open_loop}")


# ------------------------------------------------------------------------------
# q1.c
print("\n#", "-"*50)
print("q1 c) state feedback loop")

A = np.asarray(A).astype(np.float64)
print(f"A:\n{A}")

B = np.asarray(B.subs({J1: _J1, ki: _ki})).astype(np.float64)
print(f"B:\n{B}")

poles = [-2, -1, -1+1j, -1-1j]
fsf = signal.place_poles(A, B, poles, method='YT')
K = fsf.gain_matrix
print(f"K:\n{K}")

# visualizing poles 
# t = np.linspace(0, 2*np.pi, 401)
# plt.plot(np.cos(t), np.sin(t), 'k--')  # unit circle
# plt.plot(poles_open_loop.real, poles_open_loop.imag, 'rx',
#          label='open_loop eig')
# plt.plot(fsf.computed_poles.real, fsf.computed_poles.imag, 'bx',
#          label='closed_loop eig')
# plt.axis([-2.1, 1.1, -1.1, 1.1])
# plt.grid()
# plt.legend()
# plt.show()
# plt.close()

# ------------------------------------------------------------------------------
# q1 d)
print("\n#", "-"*50)
print("q1 d) simulate state feedback loop")

x_ref = np.zeros(shape=(4, 1), dtype=np.float)
x_ref[0] = 0 # phi1
x_ref[1] = 1 # phi2
print(f"x_ref:\n{x_ref}")

x = np.zeros(shape=(4, 1), dtype=np.float)
print(f"x:\n{x}")

td = 5.0
d = td / _J2
print(f"d:\n{d}\n")

C = np.array([[1, 1, 1, 1]])
print(f"C:\n{C}\n")

# A_closed-loop
print(f"BK:\n{B @ K}\n")
A_cl = A - B @ K
print(f"Closed-loop A_cl = A - BK:\n{A_cl}\n")

# dc gain
kr = -(1.0 / (np.linalg.inv(A_cl) @ B))[0, 0]
kr = kr * np.ones(shape=(1, 4))
print(f"kr: {kr}\n")
print(f"x_ref * kr:\n {x_ref * kr}")
print(f"Bkr:\n{B * kr}")

t_seconds = 10
t_step = 0.05
t_start = 0.0
t = np.arange(t_start, t_seconds, t_step)
num_iters = len(t)

Y = []
X = []

def dynamics(t, x, u):
    x = x.reshape(4, 1)
    x_dot = A_cl @ x + B * u
    return x_dot

for i in range(1, num_iters):
    u = kr @ x_ref + d
    x = x.reshape(4, )
    sol = solve_ivp(dynamics, [t[i-1], t[i]], x, args=[u], vectorized=True)
    x = sol.y[:, -1].reshape(4, 1)

    y = C @ x
    X.append(x)
    Y.append(y)

Y = np.asarray(Y)
X = np.asarray(X)
t = t[:-1]


plt.title(f'Step Response tau_d: {td}')
plt.xlabel('time')
plt.axhline(y=1.0, color='black', linestyle='--')

plt.plot(X[:, 0], label='x1')
plt.plot(X[:, 1], label='x2')
plt.plot(X[:, 2], label='dx1')
plt.plot(X[:, 3], label='dx2')

plt.legend()
plt.savefig(f'step-resp_td-{td}.png', bbox_inches='tight')
plt.show()