import cv2
import numpy as np


# Function to return empty b/w image with a small border
def get_empty_frames(img):
    empty_frame = 255 * np.ones_like(img)
    empty_frame_2D = cv2.cvtColor(empty_frame, cv2.COLOR_BGR2GRAY)
    empty_frame_2D = cv2.copyMakeBorder(empty_frame_2D, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    return empty_frame_2D


# Function to return the center coordinates and the radius of the largest area with white pixels
def DistanceTransform(empty_frame_2D):
    emp_height, emp_width = empty_frame_2D.shape

    wh = np.max([emp_height, emp_width])
    emp_temp = cv2.resize(empty_frame_2D, (wh, wh))

    distimg = cv2.distanceTransform(emp_temp, cv2.DIST_L2, 5)
    _, max_val, _, max_loc = cv2.minMaxLoc(distimg)

    EMPX_C_Temp = max_loc[0]
    EMPY_C_Temp = max_loc[1]

    EMPX_C = int(emp_width * (EMPX_C_Temp / wh))
    EMPY_C = int(emp_height * (EMPY_C_Temp / wh))
    EMP_R = int(max_val)

    return EMPX_C, EMPY_C, EMP_R
