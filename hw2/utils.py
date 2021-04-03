# This code is from the 16811 Math fundamentals for robotics class (Fall 2020)
import numpy as np
import scipy 
import random

def EstimatePlane(P):
    """
      Estimates a plane using SVD by finding the eigenvector with the lowest
      values corresponding to the eigenvalue with the lowest value as well.
      The code estimates n = (a, b, c) and d from the following equation:
                            ax + by + cz = d
      Inputs
      ------
        P: np.array containing the points for which we want to calculate a plane
      Outputs
      ------
        n: normal vector corresponding to the plane
        d: plane distance value
        avg_d: average distance from points to plane
    """
    c = np.mean(P, axis=0, dtype=np.float)
    P_c = P - c
    Pc = P_c.T @ P_c
    U, _, _ = scipy.linalg.svd(Pc)
    n = U[:, -1]
    d = np.mean(np.dot(P, n))
    avg_d = np.mean(abs(np.dot(P, n) - d))  # want: np.t - d = 0
    # print(np.mean(abs(np.dot(P, n) - d)))
    return n, d, avg_d


def RANSAC(points, n_samples=5, n_iters=100, dthresh=0.005):
    """
      Uses RANSAC to find the best model of a plane that fits a set of points.
      Inputs
      ------
        points: np.array containing the points for which we want to calculate a plane
        n_samples: Number of samples used to estimate a plane
        n_iters: Number plane estimation iterations
        d_thresh: Minimum aceptable distance to consider a point part of a model
      Outputs
      -------
        best_n: normal vector corresponding to the best estimated plane
        best_d: best plane distance value
        best_avg_d: best average distance from points to plane
    """
    N = len(points)

    # Best model parameters
    best_ninliers = 0
    best_n = [0, 0, 0]
    best_d = 0
    best_avg_d = 0
    for i in range(n_iters):
        # Get sample points
        samples = random.sample(range(N), n_samples)
        testp = points[samples]

        # Estimate a plane from sample points
        n, d, avg_d = EstimatePlane(testp)
        # Count the number of inliers
        D = abs(np.dot(points, n) - d)
        inliers = np.sum(D <= dthresh)

        if inliers > best_ninliers:
            # print(n)
            best_ninliers = inliers
            best_n = n
            best_d = d
            best_avg_d = avg_d

    return best_n, best_d, best_avg_d