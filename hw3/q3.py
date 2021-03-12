from utils import (
    Twist,
    Jacobian,
    JacobianPInv,
    JacobianInv,
    FK
)
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion
import math

from mpl_toolkits.mplot3d import Axes3D
np.set_printoptions(4, suppress=True)

# My utility functions

# ------------------------------------------------------------------------------
# Load data

# poses
# x_s = np.loadtxt("data/xs.txt", delimiter=' ', dtype=np.float)
# x_d1 = np.loadtxt("data/xd1.txt", delimiter=' ', dtype=np.float)
# x_d2 = np.loadtxt("data/xd2.txt", delimiter=' ', dtype=np.float)

# joint_data = np.loadtxt("data/joint_data.txt", delimiter=' ', dtype=np.float)
# v = np.loadtxt("data/v.txt", delimiter=' ', dtype=np.float)

# ------------------------------------------------------------------------------
# Methods


def ComputeTwists(qw_matrix):
    """
    Compute the twists corresponding to the q-w pairs.
    """
    twists = np.zeros(shape=(len(qw_matrix), 4, 4), dtype=np.float)
    for i, qw in enumerate(qw_matrix):
        q = qw[:3]
        w = qw[3:]
        twists[i] = Twist(q, w)
        print(f"Twist {i}:\n{twists[i]}")

    return twists


class IKSolver(object):

    def __init__(
        self,
        g_ssh,
        g_sht,
        twists,
        FK,
        Jacobian,
        JacobianInv,
        method = "pinv",
        K=[1, 1],
        dt=0.001,
        err_thresh=1e-2,
        max_iter=5000,
        log_step=500
    ):
        self.g_ssh = g_ssh
        self.g_sht = g_sht
        self.twists = twists
        self.FK = FK
        self.Jacobian = Jacobian
        self.JacobianInv = JacobianInv
        self.K = K
        self.dt = dt
        self.err_thresh = err_thresh
        self.max_iter = max_iter
        self.log_step = log_step
        self.method = method

        self.Q = []

    def PoseError(self, pd, qd, ps, qs):
        """
        Pose error:
            translation et = pd - ps
            orientation e0 = qd * qs^-1
        """
        e = np.zeros(shape=(6,), dtype=np.float)
        e[:3] = pd - ps
        # print(e[:3], np.linalg.norm(e[:3]))
        dq = (qd * qs.inverse)
        # e[3:] = dq.w * dq.imaginary
        e[3:] = dq.imaginary
        # print(e[3:], np.linalg.norm(e[3:]))
        return e, np.linalg.norm(e, ord=2)

    def UpdatePose(self, q):
        g = self.g_ssh @ self.FK(self.twists, q, self.g_sht)
        t = g[:3, -1]
        R = g[:3, :3]
        quat = Quaternion(matrix=R).unit
        return t, quat

    def Solve(self, x_d, q0, lamb = 0.0):
        itr = 0
        pd = x_d[:3]
        quatd = Quaternion(x_d[-1], x_d[3], x_d[4], x_d[5])
        norm_error = 10000
        
        self.Q = [q0]
        while norm_error > self.err_thresh and itr < self.max_iter:
            # compute new pose
            ps, quats = self.UpdatePose(self.Q[itr])
            v, norm_error = self.PoseError(pd, quatd, ps, quats)

            if itr % self.log_step == 0 or itr == self.max_iter - 1:
                print(f"[{itr}/{self.max_iter}]\n\tpose error: {norm_error}")
                print(f"\ttranslation: {np.linalg.norm(v[:3])}")
                print(f"\trotation: {np.linalg.norm(v[3:])}")
                print(f"\tcurrent pose: {ps}, {quats}")
                print(f"\tgoal pose: {pd}, {quatd}")
                
            # v = ek
            v[:3] *= self.K[0]
            v[3:] *= self.K[1] 

            # jacobian
            J = self.Jacobian(self.twists, self.Q[itr])
            
            J_pinv = self.JacobianInv(J, lamb) # if 0 then its the moore-penrose
            q_dot = J_pinv @ v
                

            # get new q
            q = self.Q[itr] + q_dot * self.dt
            self.Q.append(q)
            
            itr += 1    

    def Save(self, outdir="data/trajectory.txt"):
        np.savetxt(outdir, np.asarray(self.Q), fmt="%10.5f", delimiter=' ')

# ------------------------------------------------------------------------------
# Main Program


def main():
    # g spatial wrt base
    g_sb = np.loadtxt("data/g_spatial_base.txt", delimiter=" ", dtype=np.float)
    print(f"g base wrt world:\n{g_sb}")

    # g shoulder wrt base
    g_bsh = np.loadtxt("data/g_base_shoulder.txt",
                       delimiter=" ", dtype=np.float)
    print(f"g shoulder wrt base:\n{g_bsh}")

    # g tool wrt shoulder
    g_sht = np.loadtxt("data/g_shoulder_tool.txt",
                       delimiter=" ", dtype=np.float)
    print(f"G tool wrt shoulder:\n{g_sht}")

    # q and w vector to compute the twists
    qw = np.loadtxt("data/qw.txt", delimiter=' ', dtype=np.float)
    print(f"q and w vectors:\n{qw}")
    twists = ComputeTwists(qw)

    # q3.1 compute the jacobian matrix
    joint_data = np.loadtxt("data/joint_data.txt",
                            delimiter=' ', dtype=np.float)
    J = Jacobian(twists, joint_data)
    print(f"Q3.1 Jacobian:\n{J}")

    # q3.3 Jacobian pseudoinverse
    g_ssh = g_sb @ g_bsh
    g = g_ssh @ FK(twists, joint_data, g_sht)
    print(f"g_s:\n{g}")

    x_d1 = np.loadtxt("data/xd1.txt", delimiter=' ', dtype=np.float)
    IK = IKSolver(g_ssh, g_sht, twists, FK, Jacobian, JacobianInv)
    # IK.Solve(x_d1, joint_data)
    # IK.Save("data/traj_xd1.txt")
    
    # x_d2 = np.loadtxt("data/xd2.txt", delimiter=' ', dtype=np.float)
    # IK.Solve(J, x_d2, joint_data)
    # IK.Save("data/traj_xd2.txt")
    
    # q3.4 damped least squares
    IK.Solve(x_d1, joint_data, lamb=0.01)
    IK.Save("data/traj_xd1_damped.txt")


if __name__ == "__main__":
    main()
