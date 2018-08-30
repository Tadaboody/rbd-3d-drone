from new import Camera, CameraTracker
import os
import cv2 as cv
import numpy as np


BOARD_SIZE = (9, 6)
SQURESIZE = 55
def create_objp():
    objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQURESIZE
    return objp


def main():
    objp = create_objp()
    inner_calibs = [[[460, 0, 340], [0, 463, 360], [0, 0, 1]],
                    [[459, 0, 302], [0, 462, 200], [0, 0, 1]]]
    DistCoffs = [[-0.4496, 0.2395, -0.0098, -0.0010],
                 [-0.4237, 0.1677, 0.0115, -0.0002]]
    img_num = '21'
    images = ['pic{}L.png'.format(img_num), 'pic{}R.png'.format(img_num)]
    image_dir = 'pics'
    images = [os.path.join(image_dir,im) for im in images]

    def cam(i):
        return Camera(inner_calibs[i], DistCoffs[i], known_points=list(
            zip(objp, chessboard_points(images[i]))))

    tracker = CameraTracker(cam(0), cam(1))
    print(tracker.triangulate(chessboard_points(
        images[0])[0], chessboard_points(images[1])[0]))


def chessboard_points(impath):
    # Real cooridinates of the chessboard (after assuming it as the origin)
    gray = cv.imread(impath, 0)
    # gray = cv.flip(gray,-1) # Flip around y axis
    # cv.imshow("flipped",gray)
    ret_img, corners_pxls = cv.findChessboardCorners(
        gray, BOARD_SIZE, None)

    corners_pxls = np.reshape(
        corners_pxls, (BOARD_SIZE[0]*BOARD_SIZE[1], 2))

    return np.array(corners_pxls)


if __name__ == '__main__':
    main()
