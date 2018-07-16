import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



lpath1 = "files/1pic1L.png"
lpath2 = "files/pic2.png"
rpath1 = "files/1pic1R.png"
rpath2 = "files/pic2r.png"


CONST = 7.4 #TODO: const what?
def cameraCalibrate(img,inp):
    objpoints = []
    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints.append(objp*CONST)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_shape = gray.shape[::-1]
    imgpoints = []
    ret_img, corners_img = cv.findChessboardCorners(gray, (9, 6), None)
    imgpoints.append(corners_img)

    viz_frame = img.copy()
    for point in corners_img[:2]:
        cv.circle(viz_frame,tuple(map(int ,point.tolist()[0])),5,(0,255,0),-1) #TODO: convert to list comprehension

    '''
    cv.imshow("ib",viz_frame)
    if cv.waitKey(0) & 0xFF == ord('q'):
        pass
    cv.destroyAllWindows()
    '''
    #ret,rv,tv = cv.solvePnP(objp,np.array(corners_img),inp,None)
    ret_calib, CMat, DistCoff, rv, tv = cv.calibrateCamera(objpoints,imgpoints,img_shape,inp,None)

    #print "corners\n", corners_img
    #print "rv, ", cv.Rodrigues(rv[0])[0]
    #print "\ntv, ", tv

    return imgpoints, CMat, DistCoff
    #return cv.Rodrigues(rv)[0],tv

def stereoCalibrate(img1, img2,inp1,inp2):
    objpoints = []
    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints.append(objp*CONST)

    #print "obj\n", objpoints

    flags = 0
    flags |= cv.CALIB_FIX_INTRINSIC
    # flags |= cv.CALIB_FIX_PRINCIPAL_POINT
    flags |= cv.CALIB_USE_INTRINSIC_GUESS
    flags |= cv.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv.CALIB_FIX_ASPECT_RATIO
    flags |= cv.CALIB_ZERO_TANGENT_DIST
    # flags |= cv.CALIB_RATIONAL_MODEL
    # flags |= cv.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv.CALIB_FIX_K3
    # flags |= cv.CALIB_FIX_K4
    # flags |= cv.CALIB_FIX_K5
    stereocalib_criteria = (cv.TERM_CRITERIA_MAX_ITER +
        cv.TERM_CRITERIA_EPS, 100, 1e-6)


    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    shape = gray.shape[::-1]


    imgpoints_1,M1,d1 = cameraCalibrate(img1,inp1)
    imgpoints_2,M2,d2 = cameraCalibrate(img2,inp2)
    ret, M1, d1, M2, d2, R, T, E, F = cv.stereoCalibrate(
        objpoints, imgpoints_1,
        imgpoints_2, M1, d1, M2,
        d2, shape,
        criteria=stereocalib_criteria, flags=flags)
    '''
    print "m1", M1
    print "r", R
    print "t", T
    print "e", E
    print "f", F
    '''
    print("R*T \n", np.dot(R,T))
    print("R'*T \n", np.dot(np.transpose(R),T))

    return M1, d1, M2, d2, R, T

def triangultion(img1,img2):
    objpoints = []
    objp = np.zeros((9*6, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    objpoints.append(objp*CONST)
    print(objpoints)
    gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    shape = gray.shape[::-1]

    M1, d1, M2, d2, R, T = stereoCalibrate(img1, img2)
    return cv.stereoRectify(M1, d1, M2, d2, shape, R, T, alpha=-1)

def ib(img1,img2):
    inp1 = np.array([[730,0,422],[0,585,229],[0,0,1]])
    inp2 = np.array([[701,0,390],[0,562,196],[0,0,1]])


    rv1,tv1 = cameraCalibrate(img1,inp1)
    print("R1", rv1)
    print("\n R1*T1 \n", np.dot(rv1,tv1))
    rv2,tv2 = cameraCalibrate(img2,inp2)
    print("R2", rv2)
    print("\n R2*T2 \n", np.dot(rv2,tv2))
    ans = np.dot((rv1),tv1)-np.dot((rv2),tv2)
    print("\n ans:\n", ans)



def main():
    l1 = cv.imread(lpath1)
    r1 = cv.imread(rpath1)
    l2 = cv.imread(lpath2)
    r2 = cv.imread(rpath2)


    inp1 = np.array([[438.40566405, 0. ,295.56570293],
                     [0. ,443.89587156, 187.76492822],
                     [0., 0., 1.]])
    inp2 = np.array([[447.04221712, 0., 341.00865224],
                     [0., 448.39171467, 256.63590166],
                     [0., 0., 1.]])


    grayL = cv.cvtColor(l1, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(r1, cv.COLOR_BGR2GRAY)


    #ib(l1,r1)

    M1, d1, M2, d2, R, T = stereoCalibrate(l1, r1, inp1, inp2)
    #RL, RR, PL, PR, _, _, _ = triangultion(l1,r1)

# compute the pixel mappings to the rectified versions of the images
    '''
    mapL1, mapL2 = cv.initUndistortRectifyMap(M1, d1, RL, PL, grayL.shape[::-1], cv.CV_32FC1);
    mapR1, mapR2 = cv.initUndistortRectifyMap(M2, d2, RR, PR, grayL.shape[::-1], cv.CV_32FC1);


    #stereoCalibrate(l1,r1)

    #triangultion(l1,r1)


    stereoProcessor = cv.StereoSGBM_create(0, 128, 21);

    undistorted_rectifiedL = cv.remap(grayL, mapL1, mapL2, cv.INTER_LINEAR);
    undistorted_rectifiedR = cv.remap(grayR, mapR1, mapR2, cv.INTER_LINEAR);

    cv.imshow("windowNameL",undistorted_rectifiedL);
    cv.imshow("windowNameR",undistorted_rectifiedR);

    disparity = stereoProcessor.compute(undistorted_rectifiedL,undistorted_rectifiedR);
    cv.filterSpeckles(disparity, 0, 4000, 128)
    disparity_scaled = (disparity / 16.).astype(np.uint8) + abs(disparity.min())

    cv.imshow("windowNameD", disparity_scaled);

    if cv.waitKey(0) & 0xFF == ord('q'):
        pass
    cv.destroyAllWindows()

    '''
    l1 = cv.cvtColor(cv.imread(lpath1), cv.COLOR_BGR2GRAY)
    l2 = cv.cvtColor(cv.imread(lpath2), cv.COLOR_BGR2GRAY)
    r1 = cv.cvtColor(cv.imread(rpath1), cv.COLOR_BGR2GRAY)
    r2 = cv.cvtColor(cv.imread(rpath2), cv.COLOR_BGR2GRAY)

    img_shape = l1.shape[::-1]



    #print objp
    objpoints = []
    imgpoints_l = []
    imgpoints_r = []



    ret_r, corners_r = cv.findChessboardCorners(r1, (9, 6), None)

    imgpoints_l.append(corners_l)
    imgpoints_r.append(corners_l)

    '''
    '''

    #ret_calib, ClMat, DistCoff, rv, tv = cv.calibrateCamera(objpoints,imgpoints_l,img_shape,None,None)
    '''
    print "ret_calib ",ret_calib
    print "ClMat ", ClMat
    print "DistCoff ", DistCoff
    print "rv ", rv
    print "tv ", tv
    '''
    #stereo = cv.StereoBM_create(numDisparities=160, blockSize=51)
    #disparity = stereo.compute(l1,r1)
    #plt.imshow(disparity,'gray')
    #plt.show()





if __name__ == "__main__":
    main()
