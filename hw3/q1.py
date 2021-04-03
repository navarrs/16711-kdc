# -----------------------------------------------------------------------------#
# @author ingridn
# @brief  16-711 KDC Hw3 - use symbolic programing to obtain the FK kinematic
#         map (q1 a) and the body jacobian (q1 c)
# -----------------------------------------------------------------------------#
from utils import *
from q3 import ComputeTwists

from math import cos, sin
from sympy import *
init_printing(use_unicode=True)

c1 = Symbol("c1")
c2 = Symbol("c2")
c3 = Symbol("c3")
s1 = Symbol("s1")
s2 = Symbol("s2")
s3 = Symbol("s3")
l = Symbol("l")

# Matrics
exp_xi1 = Matrix([
    [c1, -s1, 0, 0],
    [s1, c1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]]
)
exp_xi2 = Matrix([
    [1, 0, 0, 0],
    [0, c2, s2, 0],
    [0, -s2, c2, 0],
    [0, 0, 0, 1]]
)

exp_xi3 = Matrix([
    [c3, 0, s3, 0],
    [0, 1, 0, 0],
    [-s3, 0, c3, 0],
    [0, 0, 0, 1]]
)

gst_0 = Matrix([[1, 0, 0, 0],
                [0, 1, 0, l],
                [0, 0, 1, 0],
                [0, 0, 0, 1]])

x1 = Matrix([0, 0, 0, 0, 0, 1])
x2 = Matrix([0, 0, 0, -c2, -s2, 0])
x3 = Matrix([0, 0, 0, -s1*c2, -c1*c2, -s2])

def ComputeAdjoint(M):
    # adjoint
    # | R^t   -R^{t}phat |
    # | 0         R^t |
    R = Matrix([
        [M[0, 0], M[0, 1], M[0, 2]],
        [M[1, 0], M[1, 1], M[1, 2]],
        [M[2, 0], M[2, 1], M[2, 2]],
    ])
    RT = R.T
    
    p_hat = Matrix([
        [0, -M[2, 3], M[1, 3]],
        [M[2, 3], 0, -M[0, 3]],
        [-M[1, 3], M[0, 3], 0]
    ])
   
    pR = -RT * p_hat

    adj = Matrix([
        [RT[0, 0], RT[0, 1], RT[0, 2], pR[0, 0], pR[0, 1], pR[0, 2]],
        [RT[1, 0], RT[1, 1], RT[1, 2], pR[1, 0], pR[1, 1], pR[1, 2]],
        [RT[2, 0], RT[2, 1], RT[2, 2], pR[2, 0], pR[2, 1], pR[2, 2]],
        [0, 0, 0, RT[0, 0], RT[0, 1], RT[0, 2]],
        [0, 0, 0, RT[1, 0], RT[1, 1], RT[1, 2]],
        [0, 0, 0, RT[2, 0], RT[2, 1], RT[2, 2]],
    ])
    return adj

def Q1a():
    print(f"FK map")
    pprint(exp_xi1 * exp_xi2 * exp_xi3 * gst_0)


def Q1c():
    eg = exp_xi3 * gst_0
    adj3 = ComputeAdjoint(eg)
    x3_dagger = adj3 * x3
    print(f"\nadjoint inv 3:\n")
    pprint(adj3)
    print(f"\nx3 dagger:\n")
    pprint(x3_dagger)

    eg = exp_xi2 * eg
    adj2 = ComputeAdjoint(eg)
    x2_dagger = adj2 * x2
    print(f"\nadjoint inv 2:\n")
    pprint(adj2)
    print(f"\nx2 dagger:\n")
    pprint(x2_dagger)

    eg = exp_xi1 * eg
    adj1 = ComputeAdjoint(eg)
    x1_dagger = adj1 * x1
    print(f"\nadjoint inv 1:\n")
    pprint(adj1)
    print(f"\nx1 dagger:\n")
    pprint(x1_dagger)

    print(f"\nBody Jacobian ")
    J_b = Matrix([
        [x1_dagger[0], x2_dagger[0], x3_dagger[0]],
        [x1_dagger[1], x2_dagger[1], x3_dagger[1]],
        [x1_dagger[2], x2_dagger[2], x3_dagger[2]],
        [x1_dagger[3], x2_dagger[3], x3_dagger[3]],
        [x1_dagger[4], x2_dagger[4], x3_dagger[4]],
        [x1_dagger[5], x2_dagger[5], x3_dagger[5]],
    ])
    pprint(J_b)

def main():

    # q1a
    Q1a()
    
    # q1c
    Q1c()


if __name__ == "__main__":
    main()
