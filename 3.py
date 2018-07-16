import cv2 as cv
import numpy as np
import os

path1 = "files/L/pic6.png"
path2 = "files/R/pic6.png"
#a1 = "files/a/pic1L.png"
#a2 = "files/a/pic1R.png"
folder1 = "files/L"
folder2 = "files/R"

CONST = 74

'''
objpoints = []
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
objpoints.append(objp*CONST)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_shape = gray.shape[::-1]
imgpoints = []
ret_img, corners_img = cv.findChessboardCorners(gray, (9, 6), None)
imgpoints.append(corners_img)
ret,rv,tv = cv.solvePnP(objp,np.array(corners_img),inp,None)
cv.Rodrigues(rv)[0],tv
'''

def innerCalib(folder):     #run once
    objpoints = []
    imgpoints = []

    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    for filename in os.listdir(folder):
        #print filename
        #print folder+"/"+filename
        img = cv.imread(folder+"/"+filename)
        objpoints.append(objp*CONST)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ret_img, corners_img = cv.findChessboardCorners(gray, (9, 6), None)
        imgpoints.append(corners_img)
    ret_calib, CMat, DistCoff, rv, tv = cv.calibrateCamera(objpoints,imgpoints,img_shape,None,None)
    print CMat
    return CMat

def PnP(impath,arr):
    objpoints = []
    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objp *= CONST
    objpoints.append(objp*CONST)

    img = cv.imread(impath)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgpoints = []
    ret_img, corners_img = cv.findChessboardCorners(gray, (9, 6), None)
    imgpoints.append(corners_img)
    ret,rv,tv = cv.solvePnP(objp,np.array(corners_img),arr,None)
    print "rt \n", cv.Rodrigues(rv)[0],tv
    return cv.Rodrigues(rv)[0],tv

def externalCalib(proj1,proj2,points1,points2):
    pts4D = cv.triangulatePoints(proj1, proj2, points1,points2)
    pts3D = np.transpose(pts4D).tolist()
    for i,point in enumerate(pts3D):
        pts3D[i] = [point[0]/point[3],point[1]/point[3],point[2]/point[3]]

    for point in pts3D:
        print point
    print "done"
    return pts3D





def main():
    #innerCalib(folder1)
    #innerCalib(folder2)
    print "a"
    arr1 = np.array([[438.40566405, 0. ,295.56570293],
                     [0. ,443.89587156, 187.76492822],
                     [0., 0., 1.]])
    arr2 = np.array([[447.04221712, 0., 341.00865224],
                     [0., 448.39171467, 256.63590166],
                     [0., 0., 1.]])

    arr1 = np.array([[479,0,303],[0,383,217],[0,0,1]])
    arr2 = np.array([[479,0,338],[0,481,234],[0,0,1]])
    rv1,tv1 = PnP(path1,arr1)
    rv2,tv2 = PnP(path2,arr2)

    proj1 = np.dot(arr1,np.concatenate((rv1,tv1),axis = 1))
    proj2 = np.dot(arr2,np.concatenate((rv2,tv2),axis = 1))
    im1 = cv.imread(path1)
    im2 = cv.imread(path2)
    gray1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    _, points1 = cv.findChessboardCorners(gray1, (9, 6), None)
    gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
    _, points2 = cv.findChessboardCorners(gray2, (9, 6), None)


    viz_frame1 = im1.copy()
    viz_frame2 = im2.copy()
    points1 = np.array([points1[0],points1[1],points1[8],points1[45],points1[-2],points1[-1]])
    points2 = [points2[0],points2[1],points2[8],points2[45],points2[-2],points2[-1]]
    for point1,point2 in zip(points1,points2):
        cv.circle(viz_frame1,tuple(map(int ,point1.tolist()[0])),5,(0,255,0),-1)
        cv.circle(viz_frame2,tuple(map(int ,point2.tolist()[0])),5,(0,255,0),-1)
    cv.imshow("1",viz_frame1)
    cv.imshow("2",viz_frame2)

    #
    print externalCalib(proj1,proj2,points1,np.array(points2))

    if cv.waitKey(0) & 0xFF == ord('q'):
        pass
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
