import cv2 as cv
import rhino
from collections import namedtuple
from typing import Tuple
import numpy as np
from numpy.linalg import inv
from Camera import Camera
from typedefs import Point3D,Line3D



class CameraTracker:
    def __init__(self, camera_1: Camera, camera_2: Camera):
        self.camera_1 = camera_1
        self.camera_2 = camera_2

    @staticmethod
    def interesct(pt11, pt12, pt21, pt12):
        def find_line(pt1, pt2):
            return [pt1,(pt2-pt1)]
        def create_plane(line):
            xy_list = [[0, 0], [0, 1], [1,0]]
            for xy in xy_list:
        def project_line(line,plane):
            rhino.
        def lines_intersection(line1,line2):
            pass
        
        line1 = findLine(pt11, pt12)
        line2 = findLine(pt21, pt22)
        palne = createPlane(line1)
        line2P = projectLine(line2, plane)


    def camera(self, i: int) -> Camera:
        return self.camera_2 if i else self.camera_1

    def triangulate(self, point2d_cam1, point2d_cam2):
        return self.interesct(self.camera(0).line_to_point(point2d_cam1), self.camera(1).line_to_point(point2d_cam2))

