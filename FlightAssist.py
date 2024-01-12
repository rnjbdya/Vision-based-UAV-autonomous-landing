import time
from dronekit import VehicleMode
from pymavlink import mavutil


def send_position(vehicle, pos_x, pos_y, pos_z):
    """
    :param vehicle: intialized vehicle
    :param pos_x: movement in x-axis (forward/backward)
    :param pos_y: movement in y-axis (left/right)
    :param pos_z: movement in z-axis (downward/upward)
    sends command for movement
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED,  # frame
        # mavlink.MAV_FRAME_LOCAL_NED, MAV_FRAME_LOCAL_OFFSET_NED,
        0b110111111000,  # type_mask
        pos_x, pos_y, pos_z,  # x, y, z positions
        0, 0, 0,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def arm_and_takeoff(vehicle, aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """
    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)
    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto
    # to prevent the command after Vehicle.simple_takeoff to execute immediately
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:  # Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)
