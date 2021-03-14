
import matplotlib.pyplot as plt
import numpy as np

from scipy.integrate import solve_ivp
from math import cos, acos, atan2, sin, pi

np.set_printoptions(4, suppress=True)

# ------------------------------------------------------------------------------
# Parameters


# ------------------------------------------------------------------------------
# Methods

def IKPlanar(L, x, radians=True):

    r = np.linalg.norm(x)

    alpha = acos((L[0]**2 + L[1]**2 - r ** 2) / (2 * L[0] * L[1]))
    theta2 = np.array([np.pi + alpha, np.pi - alpha])

    beta = acos((r**2 + L[0]**2 - L[1]**2) / (2 * L[0] * r))
    phi = atan2(x[1], x[0])
    theta1 = -(pi/2) + np.array([phi + beta, phi - beta])

    if not radians:
        theta1 *= 180 / pi
        theta2 *= 180 / pi

    return theta1, theta2


def IntDynamicsZ(t, Z, fz, m, g):
    z, dz = Z
    return dz, (fz - m * g) / m

def IntDynamicsX(t, X, fx, m):
    x, dx = X
    # return dx, (-fx - k2 * x) / d2
    return dx, fx / m

def IntDynamicsTheta(t, Theta, ftheta, I):
    theta, dtheta = Theta
    # return dtheta, (-ftheta - k3 * theta) / d3
    return dtheta, ftheta / I


class IKSolver(object):

    def __init__(
        self, L, K, D, M, I, g=9.81, dt=0.005, err_thresh=1e-3,
        max_iter=5000, log_step=100
    ):
        self.L = L
        self.K = K
        self.D = D
        self.M = M
        self.I = I
        self.g = g
        self.Mg = M * g
        self.dt = dt
        self.err_thresh = err_thresh
        self.max_iter = max_iter
        self.log_step = log_step

    def VirtualForces(self, state, z0):
        x, z, theta = state[0], state[1], state[2]
        x_dot, z_dot, theta_dot = state[3], state[4], state[5]

        fz = self.K[0] * (z0 - z) - self.D[0] * z_dot + self.Mg
        fx = -self.K[1] * x - self.D[1] * x_dot
        ftheta = -self.K[2] * theta - self.D[2] * theta_dot

        return fx, fz, ftheta

    def Jacobian(self, thetal, thetar):
        theta_la, theta_lk = thetal[0, 0], thetal[0, 1]
        theta_ra, theta_rk = thetar[1, 0], thetar[1, 1]

        A = -self.L[0] * cos(theta_la) - self.L[1] * cos(theta_la + theta_lk)
        B = -self.L[0] * sin(theta_la) - self.L[1] * sin(theta_la + theta_lk)
        C = -self.L[0] * cos(theta_ra) - self.L[1] * cos(theta_ra + theta_rk)
        D = -self.L[0] * sin(theta_ra) - self.L[1] * sin(theta_ra + theta_rk)

        Q = -self.L[1] * cos(theta_la + theta_lk)
        R = -self.L[1] * sin(theta_la + theta_lk)
        S = -self.L[1] * cos(theta_ra + theta_rk)
        T = -self.L[1] * sin(theta_ra + theta_rk)

        E = C*B - A*D
        V = Q*B - R*A
        W = S*D - T*C

        j11 = C * V / E
        j12 = D * V / E
        j13 = (-V - Q*D + R*C) / 2*E - 0.5

        j31 = -A*W / E
        j32 = -B*W / E
        j33 = (W + S*B - T*A) / 2*E - 0.5
        J = np.array([[j11, j12,  j13],
                      [0.0, 0.0, -0.5],
                      [j31, j32,  j33],
                      [0.0, 0.0, -0.5]])
        return J

    def IntegrateDynamics(self, thetal, thetar, s0, s0_dot, sd, sd_dot):

        x, z, theta = s0[0], s0[1], s0[2]
        z0 = z
        x_dot, z_dot, theta_dot = s0_dot[0], s0_dot[1], s0_dot[2]

        error = 1
        itr = 0
        states = [[x, z, theta, x_dot, z_dot, theta_dot]]
        state_des = np.array([sd[0], sd[1], sd[2], sd_dot[0], sd_dot[1], sd_dot[2]])
        
        t_start = 0.0
        print(f"Desired state: {state_des}")
        while error > self.err_thresh and itr < self.max_iter:

            if  itr % self.log_step == 0:
                print(f"Current state {itr}: {states[itr]} error: {error}")
                
            # Calculate the forces
            fx, fz, ftheta = self.VirtualForces(states[itr], z0)

            # calculate new state
            t = [t_start, t_start + self.dt]
            t_start += self.dt
            
            sol = solve_ivp(
                IntDynamicsZ, t, [z, z_dot], args=[fz, self.M, self.g])
            z, z_dot = sol.y[0, -1], sol.y[1, -1]
            
            sol = solve_ivp(
                IntDynamicsX, t, [x, x_dot], args=[fx, self.M])
            x, x_dot = sol.y[0, -1], sol.y[1, -1]
            
            sol = solve_ivp(
                IntDynamicsTheta, t, [theta, theta_dot], args=[ftheta, self.I])
            theta, theta_dot = sol.y[0, -1], sol.y[1, -1]

            states.append([x, z, theta, x_dot, z_dot, theta_dot])
            
            itr += 1
            error = np.linalg.norm(state_des - states[itr])
    
        print(f"final state {itr}: {states[-1]} error: {error}")
        return np.asarray(states), error
    
def Plot(states):
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(states[:, 0], states[:, 1], states[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('theta')
    plt.show()
    plt.savefig("data/int_dynamics.png")
    
    
# ------------------------------------------------------------------------------
# Main Program


def main():

    # q4.1
    links = np.array([0.5, 0.5])
    x_b = np.array([0.0, 0.8])
    xl_a = np.array([-0.2, 0.0])
    xr_a = np.array([0.2, 0.0])

    x_b_shifted = x_b - xl_a
    thetas_la, thetas_lk = IKPlanar(links, x_b_shifted)
    print(f"left leg:\n\tankle: {thetas_la}\n\tknee {thetas_lk}")
    thetas_la, thetas_lk = IKPlanar(links, x_b_shifted, False)
    print(f"left leg:\n\tankle: {thetas_la}\n\tknee {thetas_lk}")

    x_b_shifted = x_b - xr_a
    thetas_ra, thetas_rk = IKPlanar(links, x_b_shifted)
    print(f"right leg:\n\tankle: {thetas_ra}\n\tknee {thetas_rk}")
    thetas_ra, thetas_rk = IKPlanar(links, x_b_shifted, False)
    print(f"right leg:\n\tankle: {thetas_ra}\n\tknee {thetas_rk}")

    # q4.2
    # obtained in q4.1
    thetasl = np.array([0.3563, 1.2025])
    thetasr = np.array([0.8462, 1.2025])

    # init state (x, z, theta)
    s0 = np.array([0.0, 0.8, 0.0])
    s0_dot = np.array([0.1, -1.0, 0.1])
    sd = np.array([0.0, 0.8, 0.0])
    sd_dot = np.array([0.0, 0.0, 0.0])

    IK = IKSolver(links, K=[1000.0, 100.0, 100.0],
                  D=[300.0, 75.0, 75.0], M=80.0, I=2.0)
    states, err = IK.IntegrateDynamics(thetasl, thetasr, s0, s0_dot, sd, sd_dot)
    Plot(states)


if __name__ == "__main__":
    main()
