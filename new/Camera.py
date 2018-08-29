from typing import Tuple, List

import cv2 as cv
import numpy as np
from lazy import lazy

from typedefs import Line3D, Point3D, Point2D

inv = np.linalg.inv


class Camera:
    def __init__(self, camera_matrix: np.matrix, distCoff, descriptor_bank=None, known_points=None):
        self.camera_matrix = camera_matrix
        self.distCoff = distCoff
        self.descriptor_bank = descriptor_bank
        self.known_points: List[Tuple[Point3D, Point2D]] = known_points or list()
        self.unkown_points: List[Point2D] = list()

    def __get_image(self):
        return np.ndarray()

    def extract_points(self, image)->List[Point2D]:
        """Extracts interesting points from image"""
        BOARD_SIZE = (9, 6)
        _, corners_pxls = cv.findChessboardCorners(image, BOARD_SIZE, None)
        return corners_pxls

    def take_pic(self):
        self.known_points = None
        image = self.__get_image()
        points = self.extract_points(image)
        for point in points:
            known_point = self.descriptor_bank.get(image, point)
            if known_point:
                self.known_points.append(known_point)
            else:
                self.unkown_points.append(point)
        return image

    @property
    def location(self):
        _, tv = self.PnP(*zip(*self.known_points))
        return -tv

    def line_to_point(self, point_2d)->Line3D:
        """Returns a line between the camera to the projected points view (in the world)"""
        point3d_viewed = self.inverse_projection(point_2d)
        return self.location, point3d_viewed

    def inverse_projection(self, projected_point: np.ndarray)->np.array:
        """Given a 2d point projected on cam_i's view, return the points 3d coordinates"""
        rv, tv = self.PnP(*zip(*self.known_points))
        proj: np.ndarray = np.dot(
            self.camera_matrix, np.concatenate((rv, tv), axis=1))
        return inv(proj) @ projected_point

    def PnP(self, points3d, points2d)->Tuple[np.matrix, np.array]:
        """
        Recives points with their 3d location, and 2d projection.
        returns a rotation matrix and translation matrix of the camera in the world (6 DOF pose)
        """
        return cv.solvePnP(points3d, points2d, self.camera_matrix, self.distCoff)
