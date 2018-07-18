import cv2 as cv
import numpy as np
import os
from collections import namedtuple
path1 = "files/L/pic6.png"
path2 = "files/R/pic6.png"
#a1 = "files/a/pic1L.png"
#a2 = "files/a/pic1R.png"
folder1 = "files/L"
folder2 = "files/R"

CONST = 74 #TODO: const what?

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
        #print(  filename )
        #print(  folder+"/"+filename )
        img = cv.imread(folder+"/"+filename)
        objpoints.append(objp*CONST)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]
        ret_img, corners_img = cv.findChessboardCorners(gray, (9, 6), None)
        imgpoints.append(corners_img)
    ret_calib, CMat, DistCoff, rv, tv = cv.calibrateCamera(objpoints,imgpoints,img_shape,None,None)
    print(CMat)
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
    print("rt \n", cv.Rodrigues(rv)[0],tv)
    return cv.Rodrigues(rv)[0],tv

def externalCalib(proj1,proj2,points1,points2):
    pts4D = cv.triangulatePoints(proj1, proj2, points1,points2)
    pts3D = np.transpose(pts4D).tolist()
    for i,point in enumerate(pts3D):
        pts3D[i] = [point[0]/point[3],point[1]/point[3],point[2]/point[3]]

    for point in pts3D:
        print(point)
    print("done")
    return pts3D





GREEN = (0,255,0)
Extracted_Calibration = namedtuple(
    "ExtractedCalibration", ['proj', 'points'])  # TODO: rename

def main():
    #innerCalib(folder1)
    #innerCalib(folder2)
    print("a")

    def extract_calibration(arr : np.ndarray, path:str) -> Extracted_Calibration: #TODO: rename
        rv, tv = PnP(path, arr)
        proj = np.dot(arr, np.concatenate((rv, tv), axis=1))
        im = cv.imread(path)
        gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        _, points = cv.findChessboardCorners(gray, (9, 6), None)
        magic_indices = [0, 1, 8, 45, -2, -1]  # TODO: rename
        points = np.array([points[i] for i in magic_indices])
        draw_calib(im, points)
        return Extracted_Calibration(proj, points)

    def draw_calib(im:np.ndarray, points)->np.ndarray:
        viz_frame = im.copy()
        for point in points:
            cv.circle(viz_frame, tuple(
                map(int, point.tolist()[0])), 5, GREEN, -1)
        return viz_frame

        
    arr1 = np.array([[438.40566405, 0. ,295.56570293],
                     [0. ,443.89587156, 187.76492822],
                     [0., 0., 1.]])
    arr2 = np.array([[447.04221712, 0., 341.00865224],
                     [0., 448.39171467, 256.63590166],
                     [0., 0., 1.]])

    arr1 = np.array([[479, 0, 303], [0, 383, 217], [0, 0, 1]])
    arr2 = np.array([[479, 0, 338], [0, 481, 234], [0, 0, 1]])  # TODO: names
    arrays = [arr1, arr2]
    paths = [path1, path2]
    calibrations = [extract_calibration(arr,path) for arr,path in zip(arrays,paths)]
    
    cv.imshow("1", draw_calib(arrays[0], calibrations[0].points))
    cv.imshow("2", draw_calib(arrays[1], calibrations[1].points))

    #
    print(externalCalib(calibrations[0].proj, calibrations[1].proj, np.array(calibrations[0].point), np.array(calibrations[0].point))))

    if cv.waitKey(0) & 0xFF == ord('q'):
        pass
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
