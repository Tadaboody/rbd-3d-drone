import abc
from interfaces import Point3D
from typing import Sequence, Tuple

class Triangulator(abc.ABC):
    @abc.abstractmethod
    def __init__(self, file_paths):
        pass

    @abc.abstractmethod
    def localize(self, known_points: Sequence[Tuple[Point3D, Point3D]], unkown_points: Sequence[Point3D]) ->Sequence[Point3D]:
        pass
