import cv2 as cv
import numpy as np
import os
from collections import namedtuple
FEATURE_PARAMS = dict(  maxCorners = 250,
                        qualityLevel = 0.2,
                        minDistance = 50,
                        blockSize = 7 )
path1 = "pic1L.png"
path2 = "pic1R.png"
#a1 = "files/a/pic1L.png"
#a2 = "files/a/pic1R.png"
folder1 = "files/L"
folder2 = "files/R"

SQURESIZE = 74
GREEN = (0,255,0)

def innerCalib(folder):     #run once
    objpoints = []
    imgpoints = []

    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    for filename in os.listdir(folder):
        #print(  filename )
        #print(  folder+"/"+filename )
        img = cv.imread(folder+"/"+filename)
        objpoints.append(objp*SQURESIZE)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ret_img, corners_img = cv.findChessboardCorners(gray, (9, 6), None)
        imgpoints.append(corners_img)
    ret_calib, CMat, DistCoff, rv, tv = cv.calibrateCamera(objpoints,imgpoints,img_shape,None,None)
    print(CMat)
    return CMat


def chessboard_points(impath):
    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objp *= SQURESIZE

    img = cv.imread(impath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret_img, corners_pxls = cv.findChessboardCorners(gray, (9, 6), None)

    return objp, np.array(corners_pxls)

def PnP(real_points,screen_points,arr):
    ret,rv,tv = cv.solvePnP(real_points,screen_points,arr,None)
    print("rt \n", cv.Rodrigues(rv)[0],tv)
    return cv.Rodrigues(rv)[0],tv

def externalCalib(projMats,points):
    #import pdb; pdb.set_trace()
    pts4D = cv.triangulatePoints(projMats[0], projMats[1], np.array(points[0]),np.array(points[1]))
    pts3D = np.transpose(pts4D).tolist()
    print ("AAA",pts4D.shape)
    for i in range(len(pts3D)):
        pts3D[i] = [point/pts3D[i][3] for point in pts3D[i][:3]]

    for point in pts3D:
        print(point)
    print("done")
    return pts3D

def extract_projection_matrix(arr , path):
    rp, sp = chessboard_points(path)
    rv, tv = PnP(rp,sp, arr)
    proj = np.dot(arr, np.concatenate((rv, tv), axis=1))
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

def main():
    #innerCalib(folder1)
    #innerCalib(folder2)
    print("a")



    inner_calib1 = np.array([[438.40566405, 0. ,295.56570293],
                     [0. ,443.89587156, 187.76492822],
                     [0., 0., 1.]])
    inner_calib2 = np.array([[447.04221712, 0., 341.00865224],
                     [0., 448.39171467, 256.63590166],
                     [0., 0., 1.]])

    inner_calib1 = np.array([[479, 0, 303], [0, 383, 217], [0, 0, 1]])
    inner_calib2 = np.array([[479, 0, 338], [0, 481, 234], [0, 0, 1]])  # Inner calibrations

    inner_calibs = [inner_calib1, inner_calib2]
    paths = [path1, path2]
    images = [cv.imread(path,0) for path in paths]
    projections = [extract_projection_matrix(arr,path) for arr,path in zip(inner_calibs,paths)]
    print "Type  ", cv.goodFeaturesToTrack(cv.imread(path1,0),**FEATURE_PARAMS)
    print "Type2 ", chessboard_points(path1)[1]
    good_features = [cv.goodFeaturesToTrack(image,**FEATURE_PARAMS) for image in images]
    ch_points = [chessboard_points(path)[1] for path in paths]
    tri_points = [[frame[i] for i in [0, 1, 8, 45, -2, -1]] for frame in ch_points] #interesting points in every frame
    temp = [good_features[0][8], good_features[0][9], good_features[0][11]] #not good looking code TODO
    tri_points = [temp,good_features[1][9:12]] #interesting points in every frame
    #tri_points = good_features

    cv.imshow("1", draw_numbered_points(images[0], tri_points[0]))
    cv.imshow("2", draw_numbered_points(images[1], tri_points[1]))

    print("maz", externalCalib(projections,tri_points))
    #real dist:
        #pt 0 x:-3 y: 26, z: 0
        #pt 1 x:26 y: 10, z: 0
        #pt 2 x:84 y: 72, z: 52


    if cv.waitKey(0) & 0xFF == ord('q'):
        pass
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
