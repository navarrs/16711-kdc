# import array_to_latex as a2l
import math
import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion

# My utility functions
from utils import (
    Twist,
    Jacobian,
    JacobianPInv,
    JacobianInv,
    FK
)

np.set_printoptions(4, suppress=True)

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
        g_st, twists, FK, Jacobian, JacobianInv,
        K=[1, 1], dt=0.005, err_thresh=1e-3, max_iter=10000, log_step=1000
    ):
        self.g_st = g_st
        self.twists = twists
        self.FK = FK
        self.Jacobian = Jacobian
        self.JacobianInv = JacobianInv
        self.K = K
        self.dt = dt
        self.err_thresh = err_thresh
        self.max_iter = max_iter
        self.log_step = log_step

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
        g = self.FK(self.twists, q, self.g_st)
        t = g[:3, -1]
        R = g[:3, :3]
        quat = Quaternion(matrix=R).unit
        return t, quat

    def Solve(self, x_d, q0, lamb=0.0):
        itr = 0
        pd = x_d[:3]
        quatd = Quaternion(x_d[-1], x_d[3], x_d[4], x_d[5])
        norm_error = 100

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

            # if lambda > 0 then its damped least squares
            J_pinv = self.JacobianInv(J, lamb)
            q_dot = J_pinv @ v

            # get new q
            q = self.Q[itr] + q_dot * self.dt
            self.Q.append(q)

            itr += 1
            
        print(f"[END: {itr}]\n\tpose error: {norm_error}")
        print(f"\ttranslation: {np.linalg.norm(v[:3])}")
        print(f"\trotation: {np.linalg.norm(v[3:])}")
        print(f"\tcurrent pose: {ps}, {quats}")
        print(f"\tgoal pose: {pd}, {quatd}")

    def Save(self, outdir="data/trajectory.txt"):
        np.savetxt(outdir, np.asarray(self.Q), fmt="%10.5f", delimiter=' ')

def Q3_1(twists, joint_data):
    print(f"\n\nQ3.1", '-'*10)
    J = Jacobian(twists, joint_data)
    print(f"jacobian:\n{J}")
    # a2l.to_ltx(J, frmt='{:6.4f}', arraytype = 'array')
    
    # just a sanity check
    g_sht = np.loadtxt("data/g_shoulder_tool.txt", delimiter=' ', dtype=np.float)
    g =  FK(twists, joint_data, g_sht)
    print(f"g_s:\n{g}")
    
def Q3_2():
    print(f"\n\nQ3.2", '-'*10)
    xs = np.loadtxt("data/xs.txt", delimiter=' ', dtype=np.float)
    xd = np.loadtxt("data/xd1.txt", delimiter=' ', dtype=np.float)
    
    ps = xs[:3]
    qs = Quaternion(xs[-1], xs[3], xs[4], xs[5])
    
    pd = xd[:3]
    qd = Quaternion(xd[-1], xd[3], xd[4], xd[5])
    
    et = pd - ps
    # et = et / np.linalg.norm(et)
    eo = (qd * qs.inverse)
    eo = eo.imaginary
    
    print(f"\nv: trans {et}")
    print(f"v: orient {eo}")
    
def Q3_3(twists, joint_data):
    print(f"\n\nQ3.3", '-'*50)
    
    g_st = np.loadtxt("data/g_shoulder_tool.txt", delimiter=' ', dtype=np.float)
    xs = np.loadtxt("data/xs.txt", delimiter=' ', dtype=np.float)
    xd1 = np.loadtxt("data/xd1.txt", delimiter=' ', dtype=np.float)
    xd2 = np.loadtxt("data/xd2.txt", delimiter=' ', dtype=np.float)
    
    IK = IKSolver(g_st, twists, FK, Jacobian, JacobianInv)
    
    # solve xd1
    print(f"\nIK for pose: {xd1}")
    IK.Solve(xd1, joint_data)
    IK.Save("data/traj_xd1.txt")
    
    print(f"\nIK for pose: {xd2}")
    IK.Solve(xd2, joint_data)
    IK.Save("data/traj_xd2.txt")
    
def Q3_4(twists, joint_data):
    print(f"\n\nQ3.4", '-'*50)
    
    g_st = np.loadtxt("data/g_shoulder_tool.txt", delimiter=' ', dtype=np.float)
    xs = np.loadtxt("data/xs.txt", delimiter=' ', dtype=np.float)
    xd1 = np.loadtxt("data/xd1.txt", delimiter=' ', dtype=np.float)
    xd2 = np.loadtxt("data/xd2.txt", delimiter=' ', dtype=np.float)
    
    IK = IKSolver(g_st, twists, FK, Jacobian, JacobianInv)
    
    print(f"\nIK for pose: {xd1}")
    IK.Solve(xd1, joint_data, lamb=0.005)
    IK.Save("data/traj_xd1_damped.txt")
    
    print(f"\nIK for pose: {xd2}")
    IK.Solve(xd2, joint_data, lamb=0.005)
    IK.Save("data/traj_xd2_damped.txt")
    

# ------------------------------------------------------------------------------
# Main Program

def main():
    # parametrs ----------------------------------------------------------------
    # q and w vector to compute the twists
    qw = np.loadtxt("data/qw.txt", delimiter=' ', dtype=np.float)
    print(f"q and w vectors:\n{qw}")
    # compute the twists
    twists = ComputeTwists(qw)
    # compute jacobian
    joint_data = np.loadtxt("data/joint_data.txt", delimiter=' ', dtype=np.float)
    
    # problems -----------------------------------------------------------------
    # Q3_1(twists, joint_data)
    
    # Q3_2()
    
    # Q3_3(twists, joint_data)
    
    Q3_4(twists, joint_data)


if __name__ == "__main__":
    main()
