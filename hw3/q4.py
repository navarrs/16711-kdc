
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


def IntDynamicsZ(t, Z, az):
    z, dz = Z
    return dz, az


def IntDynamicsX(t, X, ax):
    x, dx = X
    # return dx, (-fx - k2 * x) / d2
    return dx, ax


def IntDynamicsTheta(t, Theta, atheta):
    theta, dtheta = Theta
    # return dtheta, (-ftheta - k3 * theta) / d3
    return dtheta, atheta


class IntegrateDynamics(object):

    def __init__(
        self, config, g=9.81, dt=0.005, err_thresh=1e-3,
        max_iter=5000, log_step=500
    ):
        self.L = config["links"]
        self.K = config["K"]
        self.D = config["D"]
        self.M = config["M"]
        self.I = config["I"]

        self.g = g
        self.Mg = self.M * g
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
        th_la, th_lk = thetal[0], thetal[1]
        th_ra, th_rk = thetar[0], thetar[1]

        A = -self.L[0] * cos(th_la) - self.L[1] * cos(th_la + th_lk)
        B = -self.L[0] * sin(th_la) - self.L[1] * sin(th_la + th_lk)
        C = -self.L[0] * cos(th_ra) - self.L[1] * cos(th_ra + th_rk)
        D = -self.L[0] * sin(th_ra) - self.L[1] * sin(th_ra + th_rk)

        Q = -self.L[1] * cos(th_la + th_lk)
        R = -self.L[1] * sin(th_la + th_lk)
        S = -self.L[1] * cos(th_ra + th_rk)
        T = -self.L[1] * sin(th_ra + th_rk)

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

    def Run(self, start_cond):
        
        x, z, theta = start_cond["s0"]
        z0 = z
        x_dot, z_dot, theta_dot = start_cond["s0_dot"]
        
        sd = start_cond["sd"]
        sd_dot = start_cond["sd_dot"]
        
        states = [[x, z, theta, x_dot, z_dot, theta_dot]]
        state_des = np.array(
            [sd[0], sd[1], sd[2], sd_dot[0], sd_dot[1], sd_dot[2]])
        
        # jacobian 
        J = self.Jacobian(start_cond["thetal_ak"], start_cond["thetar_ak"])

        error = 100.0
        itr = 0
        t_start = 0.0
        torques = []
        print(f"Desired state: {state_des}")
        while error > self.err_thresh and itr < self.max_iter:

            if itr % self.log_step == 0:
                print(f"Current state {itr}: {states[itr]} error: {error}")

            # forces
            fx, fz, ftheta = self.VirtualForces(states[itr], z0)
            F = np.array([fx, fz, ftheta])

            # accelerations
            az = fz / self.M - self.g
            ax = fx / self.M
            atheta = ftheta / self.I
            
            # torques
            torques.append(J @ F)

            # calculate new state
            t = [t_start, t_start + self.dt]
            t_start += self.dt

            sol = solve_ivp(IntDynamicsZ,
                            t, [z, z_dot], args=[az])
            z, z_dot = sol.y[0, -1], sol.y[1, -1]

            sol = solve_ivp(IntDynamicsX,
                            t, [x, x_dot], args=[ax])
            x, x_dot = sol.y[0, -1], sol.y[1, -1]

            sol = solve_ivp(IntDynamicsTheta,
                            t, [theta, theta_dot], args=[ftheta])
            theta, theta_dot = sol.y[0, -1], sol.y[1, -1]

            states.append([x, z, theta, x_dot, z_dot, theta_dot])

            itr += 1
            error = np.linalg.norm(state_des - states[itr])

        print(f"Final state {itr}: {states[-1]} error: {error}")
        return np.asarray(states), np.array(torques), error

def Q4_23(config, start_cond):
    IK = IntegrateDynamics(config)

    states, torques, err = IK.Run(start_cond)
    
    # plot states
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    s = start_cond["s0"]
    g = start_cond["sd"]
    
    ax.scatter3D(states[:, 0], states[:, 1], states[:, 2], '.')
    
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('theta')
    plt.savefig("data/traj_states.png")
    plt.show()
    plt.close()
    
    # plot torques 
    t = np.linspace(0, len(torques), len(torques))
    plt.plot(t, torques[:, 0], label="t_lk")
    plt.plot(t, torques[:, 1], label="t_lh")
    plt.plot(t, torques[:, 2], label="t_rk")
    plt.plot(t, torques[:, 3], label="t_rh")
    plt.xlabel("time")
    plt.ylabel("joint torque")
    plt.savefig("data/traj_torques.png")
    plt.show()
    plt.close()


def Q4_1(config):

    links = config["links"]
    xb = np.array(config["xb"])
    xl = np.array(config["xl_a"])
    xr = np.array(config["xr_a"])

    x_shifted = xb - xl
    thetas_la, thetas_lk = IKPlanar(links, x_shifted)
    print(f"left leg:\n\tankle: {thetas_la}\n\tknee {thetas_lk}")
    thetas_la, thetas_lk = IKPlanar(links, x_shifted, False)
    print(f"left leg:\n\tankle: {thetas_la}\n\tknee {thetas_lk}")

    x_shifted = xb - xr
    thetas_ra, thetas_rk = IKPlanar(links, x_shifted)
    print(f"right leg:\n\tankle: {thetas_ra}\n\tknee {thetas_rk}")
    thetas_ra, thetas_rk = IKPlanar(links, x_shifted, False)
    print(f"right leg:\n\tankle: {thetas_ra}\n\tknee {thetas_rk}")

# ------------------------------------------------------------------------------
# Main Program


def main():

    # parameters
    config = {
        "links": [0.5, 0.5],
        "K": [1000.0, 100.0, 100.0],
        "D": [300.0, 75.0, 75.0],
        "M": 80.0,
        "I": 2.0,
        "xb": [0.0, 0.8],
        "xl_a": [-0.2, 0.0],
        "xr_a": [0.2, 0.0]
    }

    # q4.1
    Q4_1(config)

    start_cond = {
        "s0": [0.0, 0.8, 0.0],
        "s0_dot": [0.1, -1.0, 0.0],
        "sd": [0.0, 0.8, 0.0],
        "sd_dot": [0.0, 0.0, 0.0],
        "thetal_ak": [0.3563, 1.2025],
        "thetar_ak": [0.8462, 1.2025]
    }

    # q4.2
    # obtained in q4.1
    Q4_23(config, start_cond)


if __name__ == "__main__":
    main()
