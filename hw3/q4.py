# -----------------------------------------------------------------------------#
# @author ingridn
# @brief  16-711 KDC Hw3 - virtual model control
# -----------------------------------------------------------------------------#
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

import numpy as np

from scipy.integrate import solve_ivp
from math import cos, acos, atan2, sin, pi

np.set_printoptions(4, suppress=True)

from utils import JacobianInv


# ------------------------------------------------------------------------------
# Methods

def IKPlanar(L, x, radians=True):

    r = np.linalg.norm(x)
    l1, l2 = L[0], L[1]
    
    # knee
    alpha = acos((l1**2 + l2**2 - r**2) / (2*l1*l2))
    # theta_knee = np.array([-pi + alpha, -pi - alpha])
    theta_knee = pi + alpha

    # ankle
    phi = atan2(x[1], x[0])
    beta = acos((r**2 + l1**2 - l2**2) / (2*l1*r))
    # theta_ankle = pi/2 - np.array([phi + beta, phi - beta])
    theta_ankle = pi/2 - phi + beta

    if not radians:
        theta_knee  *= 180 / pi
        theta_ankle *= 180 / pi

    return theta_ankle, theta_knee


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
        max_iter=5000, log_step=250
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

        l1 = self.L[0]
        l2 = self.L[1]
        
        A = -l1 * cos(th_la) - l2 * cos(th_la + th_lk)
        B = -l1 * sin(th_la) - l2 * sin(th_la + th_lk)
        C = -l1 * cos(th_ra) - l2 * cos(th_ra + th_rk)
        D = -l1 * sin(th_ra) - l2 * sin(th_ra + th_rk)

        Q = -l2 * cos(th_la + th_lk)
        R = -l2 * sin(th_la + th_lk)
        S = -l2 * cos(th_ra + th_rk)
        T = -l2 * sin(th_ra + th_rk)

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

            sol = solve_ivp(IntDynamicsZ, t, [z, z_dot], args=[az])
            z, z_dot = sol.y[0, -1], sol.y[1, -1]

            sol = solve_ivp(IntDynamicsX, t, [x, x_dot], args=[ax])
            x, x_dot = sol.y[0, -1], sol.y[1, -1]

            sol = solve_ivp(IntDynamicsTheta, t, [theta, theta_dot], args=[atheta])
            theta, theta_dot = sol.y[0, -1], sol.y[1, -1]

            states.append([x, z, theta, x_dot, z_dot, theta_dot])

            itr += 1
            error = np.linalg.norm(state_des - states[itr])

        print(f"Final state {itr}: {states[-1]} error: {error}")
        return np.asarray(states), np.array(torques), error

    def RunIK(self, start_cond):
        
        x, z, theta = start_cond["s0"]
        z0 = z
        x_dot, z_dot, theta_dot = start_cond["s0_dot"]
        
        sd = start_cond["sd"]
        sd_dot = start_cond["sd_dot"]
        
        self.states = [[x, z, theta, x_dot, z_dot, theta_dot]]
        state_des = np.array(
            [sd[0], sd[1], sd[2], sd_dot[0], sd_dot[1], sd_dot[2]])
        
        # jacobian 
        self.thetasl = [start_cond["thetal_ak"]]
        self.thetasr = [start_cond["thetar_ak"]]
        
        error = 100.0
        itr = 0
        t_start = 0.0
        torques = []
        print(f"Desired state: {state_des}")
        while error > self.err_thresh and itr < self.max_iter:

            if itr % self.log_step == 0:
                print(f"Current state {itr}: {self.states[itr]} error: {error}")

            # update J
            J = self.Jacobian(self.thetasl[itr], self.thetasr[itr])
            
            # forces
            fx, fz, ftheta = self.VirtualForces(self.states[itr], z0)
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

            sol = solve_ivp(IntDynamicsZ, t, [z, z_dot], args=[az])
            z, z_dot = sol.y[0, -1], sol.y[1, -1]

            sol = solve_ivp(IntDynamicsX, t, [x, x_dot], args=[ax])
            x, x_dot = sol.y[0, -1], sol.y[1, -1]

            sol = solve_ivp(IntDynamicsTheta, t, [theta, theta_dot], args=[atheta])
            theta, theta_dot = sol.y[0, -1], sol.y[1, -1]

            self.states.append([x, z, theta, x_dot, z_dot, theta_dot])

            # update joint angles
            X = np.array([x +0.2, z])
            theta_la, theta_lk = IKPlanar(self.L, X)  
            self.thetasl.append([theta_la, theta_lk])
            
            X = np.array([x -0.2, z])
            theta_ra, theta_rk = IKPlanar(self.L, X)  
            self.thetasr.append([theta_ra, theta_rk])
            
            itr += 1
            error = np.linalg.norm(state_des - self.states[itr])
        
        print(f"Final state {itr}: {self.states[-1]} error: {error}")
        return np.asarray(self.states), np.array(torques), error
    
    def Animate(self):
        fig, ax = plt.subplots()
        ax.axis([-0.8,0.8,0,1])
        
        # feet -- these are always fixed
        xl, xr = [-0.2, 0.0], [0.2, 0.0]
        
        def update(i):
            # state of end-effector at i
            s = self.states[i]
            
            # state of left knee
            lkx = -0.2 + self.L[0]* sin(self.thetasl[i][0])
            lkz = self.L[0] * cos(self.thetasl[i][0])
            
            # state of right knee
            rkx = 0.2 + self.L[0] * sin(self.thetasr[i][0])
            rkz = self.L[0] * cos(self.thetasr[i][0])
            
            sc = np.array([[s[0], s[1]],     # state eef
                           [lkx, lkz],       # left knee
                           [rkx, rkz],       # right knee
                           [xl[0], xl[1]],   # left ankle -- fixed
                           [xr[0], xr[1]]])  # right ankle -- fixed
            
            scatter.set_offsets(sc)
            return scatter,
        
        s = self.states[0]
        l = self.thetasl[0]
        r = self.thetasr[0]
        
        sc = np.array([[s[0], s[1]], 
                       [l[0], l[1]], 
                       [r[0], r[1]], 
                       [xl[0], xl[1]], 
                       [xr[0], xr[1]]])
        
        scatter = ax.scatter(sc[:, 0], sc[:, 1], marker='o') 
        iterations = len(self.states)
        
        ani = animation.FuncAnimation(
            fig, update, iterations, interval=50, blit=False, repeat=False)
        plt.show()
        
        ani.save('out/joints.gif', writer='pillow', fps=30)
        plt.close()


def Q4_23(config):
    
    print(f"\nQ4.2 and Q4.3 ", "-"*50)
    
    # get initial joint configuration
    links = config["links"]
    xb = np.array(config["xb"])
    xl = np.array(config["xl_a"])
    xr = np.array(config["xr_a"])
    x_shifted = xb - xl
    thetas_la, thetas_lk = IKPlanar(links, x_shifted)
    x_shifted = xb - xr
    thetas_ra, thetas_rk = IKPlanar(links, x_shifted)
    
    ID = IntegrateDynamics(config)
    
    start_cond = {
        "s0": [0.0, 0.8, 0.0],
        "s0_dot": [0.1, -1.0, 0.1],
        "sd": [0.0, 0.8, 0.0],
        "sd_dot": [0.0, 0.0, 0.0],
        "thetal_ak": [thetas_la, thetas_lk],
        "thetar_ak": [thetas_rk, thetas_rk]
    }
    states, torques, err = ID.Run(start_cond)
    
    # plot states
    fig, ax = plt.subplots(3)
    s = start_cond["s0"]
    g = start_cond["sd"]
    
    t = np.linspace(0, len(states), len(states))
    # x 
    ax[0].scatter(t[0], s[0], label=f'start: {s[0]}', s=10)
    ax[0].scatter(t[-1], g[0], label=f'goal: {g[0]}', s=10)
    ax[0].plot(t, states[:, 0], label='x')
    ax[0].legend(loc='best')
    ax[0].set(ylabel='x')
    
    # z
    ax[1].scatter(t[0], s[1], label=f'start: {s[1]}', s=10)
    ax[1].scatter(t[-1], g[1], label=f'goal: {g[1]}', s=10)
    ax[1].plot(t, states[:, 1], label='z')
    ax[1].legend(loc='best')
    ax[1].set(ylabel='z')
    
    # theta
    ax[2].scatter(t[0], s[2], label=f'start: {s[2]}', s=10)
    ax[2].scatter(t[-1], g[2], label=f'goal: {g[2]}', s=10)
    ax[2].plot(t, states[:, 2], label='theta')
    ax[2].legend(loc='best')
    ax[2].set(xlabel='time', ylabel='theta')
    
    # plt.title('states over time')
    plt.savefig("out/traj_states.png")
    plt.show()
    plt.close()
    
    # plot torques 
    t = np.linspace(0, len(torques), len(torques))
    plt.plot(t, torques[:, 0], label="t_lk")
    plt.plot(t, torques[:, 1], label="t_lh")
    plt.plot(t, torques[:, 2], label="t_rk")
    plt.plot(t, torques[:, 3], label="t_rh")
    plt.xlabel("time")
    plt.ylabel("joint torques")
    plt.legend(loc='best')
    plt.savefig("out/traj_torques.png")
    plt.show()
    plt.close()

def Q4_4(config):
    
    print(f"\nQ4.4 ", "-"*50)
    
    # get initial joint configuration
    links = config["links"]
    xb = np.array(config["xb"])
    xl = np.array(config["xl_a"])
    xr = np.array(config["xr_a"])
    x_shifted = xb - xl
    thetas_la, thetas_lk = IKPlanar(links, x_shifted)
    x_shifted = xb - xr
    thetas_ra, thetas_rk = IKPlanar(links, x_shifted)
    
    ID = IntegrateDynamics(config)
    
    start_cond = {
        "s0": [0.0, 0.8, 0.0],
        "s0_dot": [0.1, -1.0, 0.1],
        "sd": [0.0, 0.8, 0.0],
        "sd_dot": [0.0, 0.0, 0.0],
        "thetal_ak": [thetas_la, thetas_lk],
        "thetar_ak": [thetas_ra, thetas_rk]
    }
    states, torques, err = ID.RunIK(start_cond)
    
    # plot states
    s = start_cond["s0"]
    g = start_cond["sd"]
    
    # plot torques 
    t = np.linspace(0, len(torques), len(torques))
    plt.plot(t, torques[:, 0], label="t_lk")
    plt.plot(t, torques[:, 1], label="t_lh")
    plt.plot(t, torques[:, 2], label="t_rk")
    plt.plot(t, torques[:, 3], label="t_rh")
    plt.xlabel("time")
    plt.ylabel("joint torques")
    plt.legend(loc='best')
    plt.savefig("out/traj_torques_ik.png")
    plt.show()
    plt.close()
    
    ID.Animate()

def Q4_1(config):

    print(f"\nQ4.1", "-"*50)
    links = config["links"]
    xb = np.array(config["xb"])
    xl = np.array(config["xl_a"])
    xr = np.array(config["xr_a"])

    x_shifted = xb - xl
    thetas_la, thetas_lk = IKPlanar(links, x_shifted)
    print(f"\nleft leg rad:\n\tankle: {thetas_la}\n\tknee {thetas_lk}")
    thetas_la, thetas_lk = IKPlanar(links, x_shifted, False)
    print(f"left leg deg:\n\tankle: {thetas_la}\n\tknee {thetas_lk}")

    x_shifted = xb - xr
    thetas_ra, thetas_rk = IKPlanar(links, x_shifted)
    print(f"right leg rad:\n\tankle: {thetas_ra}\n\tknee {thetas_rk}")
    thetas_ra, thetas_rk = IKPlanar(links, x_shifted, False)
    print(f"right leg deg:\n\tankle: {thetas_ra}\n\tknee {thetas_rk}")

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

    # q4.2 and q4.3
    Q4_23(config)
    
    # q4.4
    Q4_4(config)


if __name__ == "__main__":
    main()
