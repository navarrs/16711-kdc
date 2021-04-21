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


# ------------------------------------------------------------------------------
# q1.a
print("\n#", "-"*50)
print("q1 a) state space model")
J1, J2, ki, k, c, s = symbols("J1,J2,ki,k,c,s")
x1, x2, x3, x4 = symbols("x1,x2,x3,x4")
k1, k2, k3, k4 = symbols("k1,k2,k3,k4")

w0 = sqrt(k * (J1 + J2) / (J1 * J2))
A = Matrix([[0.0,   0.0,       w0,      0.0],
            [0.0,   0.0,      0.0,       w0],
            [-k/J1,  k/J1, -w0*c/J1,  w0*c/J1],
            [k/J2, -k/J2,  w0*c/J2, -w0*c/J2]])
X = Matrix([x1, x2, x3, x4])
B = Matrix([0, 0, ki/J1, 0])
# distortion
B_dist = Matrix([0, 0, 0, 1/J2])
K = Matrix([k1, k2, k3, k4]).T

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
print(f"\tJ1: {_J1}\n\tJ2: {_J2}\n\tc: {_c}\n\tJ1: {_J1}\n\tk: {_k}\n\tki: {_ki}\n\tw0: {w0}")

A = A.subs({J1: _J1, J2: _J2, c: _c, k: _k, ki: _ki, w0: w0})

print("A:")
pprint(A)

print("eigen values of A:")
poles_open_loop = np.array([k for k in A.eigenvals().keys()], dtype=complex)
pprint(poles_open_loop)

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

t = np.linspace(0, 2*np.pi, 401)
plt.plot(np.cos(t), np.sin(t), 'k--')  # unit circle
plt.plot(poles_open_loop.real, poles_open_loop.imag, 'rx',
         label='open_loop eig')
plt.plot(fsf.computed_poles.real, fsf.computed_poles.imag, 'bx',
         label='closed_loop eig')
plt.axis([-2.1, 1.1, -1.1, 1.1])
plt.grid()
plt.legend()
# plt.show()
plt.close()

# ------------------------------------------------------------------------------
# q1 d)
print("\n#", "-"*50)
print("q1 d) simulate state feedback loop")

x_ref = np.zeros(shape=(4, 1), dtype=np.float)
x_ref[0] = 0.0
x_ref[1] = 1.0
print(f"x_ref:\n{x_ref}")

x = np.zeros(shape=(4, 1), dtype=np.float)
print(f"x:\n{x}")

td = 0.5
Bd = np.asarray(B_dist.subs({J2: _J2})).astype(np.float64).reshape(4, 1) * td
print(f"Bd:\n{Bd}\n")

C = np.array([1, 1, 0, 0]).reshape(4, 1)
print(f"C:\n{C}\n")

# A_closed-loop
A_cl = A - B @ K
print(f"A_closed loop:\n{A_cl}\n")

t_seconds = 10
t_step = 0.05
t_start = 0.0
t = np.arange(t_start, t_seconds, t_step)
num_iters = len(t)

Y = []
X = []

Kr = -1.0 / (C.reshape(4,) @ np.linalg.inv(A - B @ K) @ B)
print(f"Kr:\n{Kr}\n")


def dynamics(t, x, r):
    x = x.reshape(4, 1)
    x_dot = A_cl @ x + B * Kr * r + Bd
    return x_dot

for i in range(1, num_iters):
    
    x = x.reshape(4, )
    sol = solve_ivp(dynamics, [t[i-1], t[i]], x, args=[x_ref], vectorized=True)
    x = sol.y[:, -1].reshape(4, 1)

    y = C * x
    X.append(x)
    Y.append(y)

Y = np.asarray(Y)
X = np.asarray(X)
t = t[:-1]

fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)

ax[0].set_title('step response')
ax[0].plot(t, X[:, 0], label='x1')
ax[0].set_ylabel('x1')
ax[0].legend()

ax[1].plot(t, X[:, 1], label='x2')
ax[1].set_ylabel('x2')
ax[1].legend()

ax[2].plot(t, X[:, 2], label='dx1')
ax[2].set_ylabel('dx1')
ax[2].legend()

ax[3].plot(t, X[:, 3], label='dx2')
ax[3].set_ylabel('dx2')
ax[3].legend()

plt.xlabel('t')
plt.legend()
plt.show()
