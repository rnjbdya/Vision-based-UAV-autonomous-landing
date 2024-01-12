import math
import time
import numpy as np

from dronekit import VehicleMode

import Pid
from FlightAssist import send_position


class ControlDrone:
    def __init__(self, vehicle, hres=640, vres=480, hfov=74, vfov=62,
                 kp=0.5, ki=0.001, kd=0.01, clearance_m=1):
        self.vehicle = vehicle
        self.hres = hres
        self.vres = vres
        self.hfov = hfov
        self.vfov = vfov
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.clearance_m = clearance_m

        self.x_pid = Pid.pid(self.kp, self.ki, self.kd)
        self.y_pid = Pid.pid(self.kp, self.ki, self.kd)

    def pixels_per_meter(self, alt):
        PM_X = (alt * math.tan(math.radians(self.hfov / 2))) / (self.hres / 2)
        PM_Y = (alt * math.tan(math.radians(self.vfov / 2))) / (self.vres / 2)
        return PM_X, PM_Y

    def move_to_heli(self, EX_P, EY_P, HXMIN_P, HYMIN_P, HXMAX_P, HYMAX_P, obstacles, obs_unsafe_time):
        if self.vehicle.location.global_relative_frame.alt <= 1.5:
            self.vehicle.mode = VehicleMode('LAND')
            print('Landing complete')
            return None, None, None, None, None, None, None, None
        elif EX_P is not None:
            alt = self.vehicle.location.global_relative_frame.alt

            dt_x = self.x_pid.get_dt(1)
            dt_y = self.y_pid.get_dt(1)

            # Control Values in pixels
            OX_P = self.x_pid.get_pid(EX_P, dt_x)
            OY_P = self.y_pid.get_pid(EY_P, dt_y)

            PM_X, PM_Y = self.pixels_per_meter(alt)

            DLR_M = PM_X * OX_P
            DFB_M = PM_Y * OY_P

            EX_M = PM_X * EX_P
            EY_M = PM_Y * EY_P
            if math.sqrt(EX_M ** 2 + EY_M ** 2) > 2:
                DUD_M = 0
            else:
                if obstacles is not None:
                    # obstacle clearance threshold in pixels for x and y coordinates
                    x_clearance_p = self.clearance_m / PM_X
                    y_clearance_p = self.clearance_m / PM_Y

                    x_min_threshold = HXMIN_P - x_clearance_p
                    x_max_threshold = HXMAX_P + x_clearance_p
                    y_min_threshold = HYMIN_P - y_clearance_p
                    y_max_threshold = HYMAX_P + y_clearance_p

                    mask_xmin = obstacles[:, 2] < x_min_threshold
                    mask_xmax = obstacles[:, 0] > x_max_threshold
                    mask_ymin = obstacles[:, 3] < y_min_threshold
                    mask_ymax = obstacles[:, 1] > y_max_threshold

                    mask_xy = np.logical_not(
                        np.array(mask_xmin) + np.array(mask_xmax) + np.array(mask_ymin) + np.array(mask_ymax))

                    # if len(obstacles_within_danger_zone[mask]) == 0:
                    if np.any(mask_xy):
                        if obs_unsafe_time is None:
                            obs_unsafe_time = time.time()
                        DUD_M = 0
                        print('Obstacle present in the danger zone')
                    else:
                        if obs_unsafe_time is not None:
                            obs_unsafe_time = None
                        DUD_M = 0.2
                        print('Obstacle present but not in danger zone')
                        print('Lowering the drone by 0.2 metre')
                else:
                    if obs_unsafe_time is not None:
                        obs_unsafe_time = None
                    DUD_M = 0.2
                    print('No obstacle present. Lowering the drone by 0.2 metre')

            send_position(self.vehicle, DFB_M, DLR_M, DUD_M)
            return OX_P, OY_P, DLR_M, DFB_M, self.kp, self.ki, self.kd, alt, obs_unsafe_time

        elif self.vehicle.location.global_relative_frame.alt > 30:
            self.vehicle.mode = VehicleMode('LAND')
            return None, None, None, None, None, None, None, None, None
        else:
            send_position(self.vehicle, 0, 0, -0.25)
            return None, None, None, None, None, None, None, None, None

    def move_to_empty(self, EX_P, EY_P, RAD_P):
        if self.vehicle.location.global_relative_frame.alt <= 1.5:
            self.vehicle.mode = VehicleMode('LAND')
            print('Landing complete')
            return None, None, None, None, None, None, None, None
        elif EX_P is not None:
            alt = self.vehicle.location.global_relative_frame.alt
            PM_X, PM_Y = self.pixels_per_meter(alt)

            RAD_M = PM_X * RAD_P

            dt_x = self.x_pid.get_dt(1)
            dt_y = self.y_pid.get_dt(1)

            # Control Values in pixels
            OX_P = self.x_pid.get_pid(EX_P, dt_x)
            OY_P = self.y_pid.get_pid(EY_P, dt_y)

            DLR_M = PM_X * OX_P
            DFB_M = PM_Y * OY_P

            EX_M = PM_X * EX_P
            EY_M = PM_Y * EY_P
            if math.sqrt(EX_M ** 2 + EY_M ** 2) > 2:
                DUD_M = 0
            else:
                # at altitude of 3.5 m the area visible on camera is 4.0414m vertically and horizantally
                if alt > 3.5 and RAD_M > 2:
                    print('No obstacle present. Lowering the drone by 0.2 metre')
                    DUD_M = 0.2
                elif alt <= 3.5:
                    DUD_M = 0.2
                else:
                    DUD_M = 0
            send_position(self.vehicle, DFB_M, DLR_M, DUD_M)
            return OX_P, OY_P, DLR_M, DFB_M, self.kp, self.ki, self.kd, alt

        elif self.vehicle.location.global_relative_frame.alt > 30:
            self.vehicle.mode = VehicleMode('LAND')
            return None, None, None, None, None, None, None, None
        else:
            send_position(self.vehicle, 0, 0, -0.25)
            return None, None, None, None, None, None, None, None
