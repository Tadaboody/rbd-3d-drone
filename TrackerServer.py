import numpy as np


def triangulate(cam1_points, cam2_points, inner_calibrations, rts, tvs):
    for p1, p2 in zip(cam1_points, cam2_points):
        yield compute_location(p1, p2, inner_calibrations[0], rts[0], tvs[0], inner_calibrations[1], rts[1], tvs[1])


def compute_location(cam1_point, cam2_point, K1, R1, T1, K2, R2, T2):
    M = np.stack([cam1_point, cam2_point])
    d1 = np.array([M[0][0], M[0][1], K1[0][0]])
    temp = np.array([ K1[0][2], K1[1][2], 0 ])
    d1 -= temp
    d1 = R1 @ d1  # .transpose()

    d2 = np.array([M[1][0], M[1][1], K2[0][0]])
    temp = np.array([K2[0][2], K2[1][2], 0])
    d2 -= temp
    d2 = R2 @ d2
    result = min_dis_point(R1@T1, d1, R2@T2, d2)
    result = (-result[1], result[0], result[2])
    return result


def min_dis_point(a, d1, b, d2):
    a = a.reshape((3,))  # make them row vectors
    b = b.reshape((3,))
    w = np.cross(d1, d2)
    wdotw = np.dot(w, w)
    bminusa = b - a
    p = a + (np.cross(bminusa, d2).dot(w)/wdotw)*d1
    q = b + (np.cross(bminusa, d1).dot(w)/wdotw)*d2
    m = (p+q)/2
    return m
