import json
import os
from collections import namedtuple

import cv2 as cv
import numpy as np

FEATURE_PARAMS = dict(  maxCorners = 250,
                        qualityLevel = 0.2,
                        minDistance = 50,
                        blockSize = 7 )
path1 = "pic1L.png"
path2 = "pic1R.png"
BOARD_SIZE = (9,6)
# path1 = path2 = "pic1R.png"
# BOARD_SIZE = (7,7)
# BOARD_SIZE = (9,6)
#a1 = "files/a/pic1L.png"
#a2 = "files/a/pic1R.png"
folder1 = "files/L"
folder2 = "files/R"

SQURESIZE = 74
GREEN = (0,255,0)

def innerCalib(folder):     #run once
    objpoints = []
    imgpoints = []

    objp = create_objp()
    for filename in os.listdir(folder):
        #print(  filename )
        #print(  folder+"/"+filename )
        img = cv.imread(folder+"/"+filename)
        objpoints.append(objp*SQURESIZE)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ret_img, corners_img = cv.findChessboardCorners(gray, BOARD_SIZE, None)
        imgpoints.append(corners_img)
    ret_calib, CMat, DistCoff, rv, tv = cv.calibrateCamera(objpoints,imgpoints,img_shape,None,None)
    print(CMat)
    return CMat

def chessboard_points(impath):
    # Real cooridinates of the chessboard (after assuming it as the origin)
    gray = cv.imread(impath,0)
    # gray = cv.flip(gray,1) # Flip around y axis
    # cv.imshow("flipped",gray)
    ret_img, corners_pxls = cv.findChessboardCorners(gray, BOARD_SIZE, None)

    corners_pxls = np.reshape(corners_pxls,(BOARD_SIZE[0]*BOARD_SIZE[1],2))

    return np.array(corners_pxls)

def create_objp():
    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQURESIZE
    return objp

def PnP(real_points,screen_points,inner_calibration):
    ret, rv, tv = cv.solvePnP(
        real_points, screen_points, inner_calibration, None)
    print("rt \n", cv.Rodrigues(rv)[0],tv)
    rv = cv.Rodrigues(rv)[0]
    return rv,tv

def add_one_column(li):
    def new_point(point):
        return point[0], point[1], 1
    return np.array([new_point(point) for point in li])

def ib():
    used_path = path2
    num_camera = 1
    inner_calibration = load_inner_calib()[num_camera]
    objp = add_one_column(create_objp())
    proj = extract_projection_matrix(inner_calibration,used_path )
    n_3 = proj @ np.array(objp).T
    def de_hom(point):
        return point[:2] / point[2]
    n_3 = n_3.T
    n_3 = [de_hom(point) for point in n_3] 
    cv.imshow("show",draw_numbered_points(cv.imread(used_path),n_3))
    cv.imshow("real", draw_numbered_points(cv.imread(used_path),
                                           chessboard_points(used_path)[:51]))
    r,t = PnP(create_objp(),chessboard_points(used_path),inner_calibration)
    r2,t2 = PnP(create_objp(),chessboard_points(path1),load_inner_calib()[0])

    print(np.linalg.norm(t-t2))
    if cv.waitKey(0) & 0xFF == ord('q'):
        pass
    cv.destroyAllWindows()

def ib_boogaloo():
    # inv = np.linalg.inv
    inner_calibration1,inner_calibration2 = load_inner_calib()
    # path2 = path1
    # inner_calibration2 = inner_calibration1

    r1,t1 = PnP(create_objp(),chessboard_points(path1),inner_calibration1)
    r2,t2 = PnP(create_objp(),chessboard_points(path2),inner_calibration2)
    world_points = create_objp()
    chess_points1 = chessboard_points(path1)
    chess_points2 = chessboard_points(path2)
    # res = ( r.T @ add_one_column(chessboard_points(path)) ).T @ inv(inner_calibration) - r.T @ t
    locs = {key: val.tolist() for key, val in locals().items()}
    with open("locals.json",'w') as fp:
        json.dump(locs,fp)
        

def externalCalib(projMats,points):
    pts4D = cv.triangulatePoints(projMats[0], projMats[1], np.array(points[0]),np.array(points[1]))
    pts3D = np.transpose(pts4D).tolist()
    print ("AAA",pts4D.shape)
    for i in range(len(pts3D)):
        pts3D[i] = [point/pts3D[i][3] for point in pts3D[i][:3]]

    for point in pts3D:
        print(point)
    print("done")
    return pts3D

def extract_projection_matrix(inner_calib , path):
    sp = chessboard_points(path)
    rp = create_objp()
    rv, tv = PnP(rp,sp, inner_calib)
    proj = np.dot(inner_calib, np.concatenate((rv, tv), axis=1))
    return proj

def draw_numbered_points(im, points):
    viz_frame = im.copy()
    print("ZAM", viz_frame.shape)
    for i,point in enumerate(points):
        point = np.array(point).flatten()
        cv.circle(viz_frame, tuple(
            map(int, point.tolist())), 5, GREEN, -1)
        #print ("point", point)
        cv.putText(viz_frame,str(i),tuple(int(i) for i in point.flatten()),cv.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255))
        #cv.putText(viz_frame,'OpenCV',(10,500), cv.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),2,cv.LINE_AA)
    return viz_frame

def new_main():
    inner_calib1, inner_calib2 = load_inner_calib()
    inner_calibs = [inner_calib1, inner_calib2]
    paths = [path1, path2]
    images = [cv.imread(path,0) for path in paths]

    corners = [chessboard_points(path) for path in paths]
    pnps = [PnP(create_objp(), corner, inner_calib)
            for corner, inner_calib in zip(corners, inner_calibs)]
    
    rv1 = pnps[0][0]
    rv2 = pnps[1][0]
    tv1 = pnps[0][1]
    tv2 = pnps[1][1]
    ib = np.matmul(rv1,tv1)-np.matmul(rv2,tv2) 
    print("ib", np.linalg.norm(ib))
    return


def main():


    inner_calib1, inner_calib2 = load_inner_calib()
    inner_calibs = [inner_calib1, inner_calib2]
    paths = [path1, path2]
    images = [cv.imread(path,0) for path in paths]



    projections = [extract_projection_matrix(arr,path) for arr,path in zip(inner_calibs,paths)]
    PnPs = [PnP(create_objp(),chessboard_points(path),inner_calib) for path,inner_calib in zip(paths,inner_calibs)]
    # print "Type  ", cv.goodFeaturesToTrack(cv.imread(path1,0),**FEATURE_PARAMS)
    # print "Type2 ", chessboard_points(path1)[1]
    rv1 = PnPs[0][0]
    rv2 = PnPs[1][0]
    tv1 = PnPs[0][1]
    tv2 = PnPs[1][1]
    wp = create_objp()
    print("IB", np.matmul(PnPs[0][0], PnPs[0][1]) -
          np.matmul(PnPs[1][0], PnPs[1][1]))
    print("IBNORM", np.linalg.norm(rv1 @ tv1 - rv2 @ tv2))
    good_features = [cv.goodFeaturesToTrack(image,**FEATURE_PARAMS) for image in images]
    ch_points = [chessboard_points(path) for path in paths]
    tri_points = [[frame[i] for i in [0, 1, 8, 45, -2, -1]] for frame in ch_points] #interesting points in every frame
    temp = [good_features[0][8], good_features[0][9], good_features[0][11]] #not good looking code TODO
    # tri_points = [temp,good_features[1][9:12]] #interesting points in every frame
    #tri_points = good_features

    cv.imshow("1", draw_numbered_points(images[0], tri_points[0]))
    # cv.imshow("2", draw_numbered_points(images[1], tri_points[1]))
    def translate_point(point):
        point[0] += 270
        point[1] += 180
        return point[:2]
    moved_objp = np.array([translate_point(point) for point in create_objp()])
    cv.imshow("1", draw_numbered_points(images[0], tri_points[0]))
    vs_img = cv.imshow("world",draw_numbered_points(images[0],moved_objp))
    # print("maz", externalCalib(projections,tri_points))
    #real dist:
        #pt 0 x:-3 y: 26, z: 0
        #pt 1 x:26 y: 10, z: 0
        #pt 2 x:84 y: 72, z: 52


    if cv.waitKey(0) & 0xFF == ord('q'):
        pass
    cv.destroyAllWindows()

def load_inner_calib():
    JSON_PATH = "inner_calibrations.json"
    try:
        with open(JSON_PATH) as fp:
            inner_calib1, inner_calib2 = np.array(json.load(fp))
    except FileNotFoundError:
        inner_calib = [innerCalib(folder1), innerCalib(folder2)]
        with open(JSON_PATH, 'w') as fp:
            json.dump(inner_calib, fp)
    # inner_calib1 = np.array([[438.40566405, 0., 295.56570293],
    #                          [0., 443.89587156, 187.76492822],
    #                          [0., 0., 1.]])
    # inner_calib2 = np.array([[447.04221712, 0., 341.00865224],
    #                          [0., 448.39171467, 256.63590166],
    #                          [0., 0., 1.]])

    # inner_calib1 = np.array([[479, 0, 303], [0, 383, 217], [0, 0, 1]])
    # inner_calib2 = np.array([[479, 0, 338], [0, 481, 234], [0, 0, 1]])  # Inner calibrations

    return inner_calib1, inner_calib2

if __name__ == "__main__":
    ib_boogaloo()
