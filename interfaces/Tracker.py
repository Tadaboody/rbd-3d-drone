from interfaces import Point3D, Image
import abc
from typing import Sequence


class Tracker(abc.ABC):
    """Interface for tracking the """ 
    tracked_points: Sequence[Point3D]

    @abc.abstractmethod
    def localize(self, image: Image) -> Sequence[Point3D]:
        """When given the image from a drones POV. returns the drones location in the 3d world"""
        pass
