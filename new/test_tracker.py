from new import CameraTracker 
import numpy as np

def test_intersect():
    pt11 = np.array((0, 2, -1), dtype=np.float32)
    pt12 = np.array((1, 3, 1), dtype=np.float32)
    pt21 = np.array((1, 0, -1), dtype=np.float32)
    pt22 = np.array((2, 1, 2), dtype=np.float32)
    inters = CameraTracker.interesct(pt11, pt12, pt21, pt22)
    assert all(x==y for x,y in zip(inters,np.array((-1.5, 0.5, -4))))
