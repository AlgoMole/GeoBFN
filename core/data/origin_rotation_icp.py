import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from time import time


def best_fit_transform(A, B):
    """
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def get_assignments(src, dst):
    distance_mtx = cdist(src, dst, metric="euclidean")
    _, dest_ind = linear_sum_assignment(distance_mtx, maximize=False)
    distances = distance_mtx[range(len(dest_ind)), dest_ind]
    return distances, dest_ind


def icp(A, B, max_iterations=50, tolerance=0.001):
    """
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        R: final Rotation matrix for A
        rotated: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    """

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    src = np.copy(A)
    dst = np.copy(B)

    prev_error = 0
    iter_count = 0
    for i in range(max_iterations):
        # get assignments
        distances, indices = get_assignments(src, dst)

        # compute the transformation between the current source and nearest destination points
        _, R, _ = best_fit_transform(src, dst[indices, :])

        # rotate and update the current source
        src = np.dot(R, src.T).T

        # check error
        mean_error = np.max(distances)
        iter_count += 1
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    if i > max_iterations - 1:
        print("out of iteration")

    # calculate final transformation
    _, R, _ = best_fit_transform(A, src)
    _A_rotated = np.dot(R, A.T).T
    _, A_indices = get_assignments(B, _A_rotated)
    A_rotated = _A_rotated[A_indices, :]
    return R, A_rotated, A_indices, iter_count
