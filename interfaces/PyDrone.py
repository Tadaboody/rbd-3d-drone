import abc
import numpy
from interfaces import Point3D, Tracker
from typing import ClassVar

class PyDrone(abc.ABC):
    location: Point3D
    dest: Point3D
    tracker: Tracker

    def __init__(self, starting_location: Point3D, dest: Point3D, tracker_cls: ClassVar):
        self.location = starting_location
        self.tracker = tracker_cls()
        self.dest = dest
        self.flying = True

    @abc.abstractmethod
    def take_picture(self) ->numpy.ndarray:
        """Takes a picture from the drones POV"""
        pass

    @abc.abstractmethod
    def fly(self, origin: Point3D, dest: Point3D):
        """Controls the drone to fly from origin to dest"""
        pass

    def run(self):
        """Drone main loop"""
        while self.flying:
            self.location = self.tracker.localize(self.take_picture())
            self.fly(self.location, self.dest)
