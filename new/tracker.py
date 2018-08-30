import numpy as np
from new import Camera,Point3D,Line3D



class CameraTracker:
    def __init__(self, camera_1: Camera, camera_2: Camera):
        self.camera_1 = camera_1
        self.camera_2 = camera_2

    @staticmethod
    def interesct(pt11, pt12, pt21, pt22):
        pts=[0,pt11,pt12,pt21,pt22]
        assert all(p.shape == (3,) for p in pts[1:])
        def d(m,n,o,p):
            """
            dmnop = (xm - xn)(xo - xp) + (ym - yn)(yo - yp) + (zm - zn)(zo - zp)
            """
            

            def sing(i):
                return (pts[m][i]-pts[n][i])*(pts[o][i]-pts[p][i])
            
            return float(sum(sing(i) for i in range(3)))
            
        #mua = ( d1343 d4321 - d1321 d4343 ) / ( d2121 d4343 - d4321 d4321 )
        mua = (d(1, 3, 4, 3) * d(4, 3, 2, 1) - d(1, 3, 2, 1)*d(4, 3, 4, 3)) / \
            (d(2, 1, 2, 1) * d(4, 3, 4, 3) - d(4, 3, 2, 1) * d(4, 3, 2, 1))
        return pt11 + mua*(pt12-pt11)



    def camera(self, i: int) -> Camera:
        return self.camera_2 if i else self.camera_1

    def triangulate(self, point2d_cam1, point2d_cam2):
        return self.interesct(*self.camera(0).line_to_point(point2d_cam1), *self.camera(1).line_to_point(point2d_cam2))
