# Opencv Imports
import cv2
import numpy as np

# Python Imports
import math
import time
import queue, threading

# Dronekit Imports
from dronekit import LocationGlobalRelative

# Common Library Imports
from PositionVector import PositionVector


class SimulateLandingPad:
    def __init__(self, filename, vehicle, vehicleAttitude=0, backgroundColor=(74, 88, 109), target_size=1.5,
                 camera_width=640, camera_height=480, vfov=60, hfov=60, frame_rate=30):
        self.filename = filename
        self.vehicle = vehicle
        self.targetLocation = PositionVector()
        self.vehicleLocation = PositionVector()
        self.vehicleAttitude = vehicleAttitude
        self.backgroundColor = backgroundColor
        self.target_size = target_size
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.vfov = vfov
        self.hfov = hfov
        self.camera_fov = math.sqrt(self.vfov ** 2 + self.hfov ** 2)
        self.frame_rate = frame_rate
        self.current_milli_time = lambda: int(round(time.time() * 1000))

        self.end_the_thread = False
        self.__init__heli()
        self.q = queue.Queue(maxsize=4)

        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def __init__heli(self):
        self.target = cv2.imread(self.filename)
        self.target_width = self.target.shape[1]
        self.target_height = self.target.shape[0]
        self.pixels_per_meter = (self.target_height + self.target_width) / (2.0 * self.target_size)
        print("Target loaded.")
        self.target_location = LocationGlobalRelative(self.vehicle.location.global_relative_frame.lat + 0.00002,
                                                      self.vehicle.location.global_relative_frame.lon - 0.00002,
                                                      self.vehicle.location.global_relative_frame.alt)
        self.set_target_location(self.target_location)
        print("Target set.")

    def set_target_location(self, location):
        self.targetLocation.set_from_location(location)

    def project_3D_to_2D(self, thetaX, thetaY, thetaZ, aX, aY, aZ, cX, cY, cZ):
        dX = math.cos(-thetaY) * (math.sin(-thetaZ) * (cY - aY) + math.cos(-thetaZ) * (cX - aX)) \
             - math.sin(-thetaY) * (aZ - cZ)

        dY = math.sin(-thetaX) * (math.cos(-thetaY) * (aZ - cZ) + math.sin(-thetaY) *
                                  (math.sin(-thetaZ) * (cY - aY) + math.cos(-thetaZ) * (cX - aX))) + \
             math.cos(-thetaX) * (math.cos(-thetaZ) * (cY - aY) - math.sin(-thetaZ) * (cX - aX))

        dZ = math.cos(-thetaX) * (math.cos(-thetaY) * (aZ - cZ) + math.sin(-thetaY) * (
                    math.sin(-thetaZ) * (cY - aY) + math.cos(-thetaZ) * (cX - aX))) - math.sin(-thetaX) * (
                         math.cos(-thetaZ) * (cY - aY) - math.sin(-thetaZ) * (cX - aX))
        eX = 0
        eY = 0
        eZ = 1.0 / math.tan(math.radians(self.camera_fov) / 2.0)
        bX = (dX - eX) * (eZ / dZ)
        bY = (dY - eY) * (eZ / dZ)
        sX = bX * self.camera_width
        sY = bY * self.camera_height
        return sX, sY

    def shift_to_image(self, pt, width, height):
        return (pt[0] + width / 2), (-1 * pt[1] + height / 2.0)

    def simulate_target(self, thetaX, thetaY, thetaZ, aX, aY, aZ, cX, cY, cZ):
        img_width = self.target.shape[1]
        img_height = self.target.shape[0]
        corners = np.float32([[-img_width / 2, img_height / 2], [img_width / 2, img_height / 2],
                              [-img_width / 2, -img_height / 2], [img_width / 2, -img_height / 2]])
        newCorners = np.float32([[0, 0], [0, 0], [0, 0], [0, 0]])
        for i in range(0, len(corners)):
            x = corners[i][0] + cX - img_width / 2.0
            y = corners[i][1] + cY - img_height / 2.0
            x, y = self.project_3D_to_2D(thetaX, thetaY, thetaZ, aY, aX, aZ, y, x, cZ)
            x, y = self.shift_to_image((x, y), self.camera_width, self.camera_height)
            newCorners[i] = x, y
        M = cv2.getPerspectiveTransform(corners, newCorners)
        sim = cv2.warpPerspective(self.target, M, (640, 480), borderValue=(74, 88, 109))
        return sim

    def get_frame(self):
        start = self.current_milli_time()
        # print('Inside get_frame', self.targetLocation, self.vehicleLocation, self.vehicleAttitude, self.pixels_per_meter)
        aX, aY, aZ = self.targetLocation.x, self.targetLocation.y, self.targetLocation.z
        cX, cY, cZ = self.vehicleLocation.x, self.vehicleLocation.y, self.vehicleLocation.z
        thetaX = self.vehicleAttitude.pitch
        thetaY = self.vehicleAttitude.roll
        thetaZ = self.vehicleAttitude.yaw

        aX = aX * self.pixels_per_meter
        aY = aY * self.pixels_per_meter
        aZ = aZ * self.pixels_per_meter
        cX = cX * self.pixels_per_meter
        cY = cY * self.pixels_per_meter
        cZ = cZ * self.pixels_per_meter

        sim = self.simulate_target(thetaX, thetaY, thetaZ, aX, aY, aZ, cX, cY, cZ)

        while (1000/self.frame_rate) > (self.current_milli_time() - start):
            pass
        return sim

    def refresh_simulator(self, vehicleLoc, vehicleAtt):
        self.vehicleLocation.set_from_location(vehicleLoc)
        self.vehicleAttitude = vehicleAtt

    def _reader(self):
        try:
            while not self.end_the_thread:
                location = self.vehicle.location.global_relative_frame
                attitude = self.vehicle.attitude
                self.refresh_simulator(location, attitude)
                frame = self.get_frame()
                if not self.q.empty():
                    self.q.get_nowait()

                self.q.put((True, frame))
                time.sleep(0.1)
        except Exception as e:
            print(e)

    def read(self):
        if not self.q.empty():
            return self.q.get()
        else:
            return False, None

    def release(self):
        self.end_the_thread = True
