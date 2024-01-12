import os
import sys
import cv2
import time
import torch
import datetime
import argparse
import numpy as np
import pandas as pd
from dronekit import connect, VehicleMode

from models.experimental import attempt_load
from utils.general import non_max_suppression

from Control import ControlDrone
from Sim import SimulateLandingPad
from FlightAssist import arm_and_takeoff
from DistanceTransform import get_empty_frames, DistanceTransform

"""Deep Sort Imports"""
from utils_ds.parser import get_config
from deep_sort import build_tracker
"""Deep Sort End"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finding Helipad after mission complete and performing simple landing')
    parser.add_argument('--source', default='rs', help='Either Intel Realsense: rs; or Raspberry pi-cam2: pi')
    parser.add_argument('--capture_width', default=640)
    parser.add_argument('--capture_height', default=480)
    parser.add_argument('--detection_width', default=640)
    parser.add_argument('--detection_height', default=480)
    parser.add_argument('--frame_rate', default=60)
    parser.add_argument('--flip_method', default=0)
    parser.add_argument('--hfov', default=74, help='horizontal field of view of the camera')
    parser.add_argument('--vfov', default=62, help='vertical field of view of the camera')
    parser.add_argument('--kp', default=0.5,)
    parser.add_argument('--ki', default=0.001)
    parser.add_argument('--kd', default=0.01)
    parser.add_argument('--clearance', default=1, help='clearance required between landing point and obstacles in metre')
    parser.add_argument('--deepsort_config', default="./configs/deep_sort.yaml", help="Path to deepsort config file")
    parser.add_argument('--model_weight', default="./weights/for_licensing/best.pt", help="Path to YOLOv5 model weights")
    parser.add_argument('--device', default="cuda", help="Whether to use CUDA or CPU")
    args = parser.parse_args()

    # loading YOLOv5
    use_cuda = args.device
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda == 'cuda') else "cpu")
    Object_classes = ['people', 'vehicle', 'helipad', 'tree']
    Object_colors = list(np.random.rand(80, 3) * 255)
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    yolo_model = attempt_load(weights=args.model_weight, device=device)

    # Build DeepSORT Tracker
    cfg = get_config()
    cfg.merge_from_file(args.deepsort_config)
    deepsort = build_tracker(cfg, use_cuda=torch.cuda.is_available())

    # Parameters for frames capture and object detection
    capture_width = args.capture_width
    capture_height = args.capture_height
    detection_width = args.detection_width
    detection_height = args.detection_height
    frame_rate = args.frame_rate
    flip_method = args.flip_method
    source = args.source
    hfov = args.hfov
    vfov = args.vfov

    # Parameters for PID controller
    kp = args.kp
    ki = args.ki
    kd = args.kd
    clearance_m = args.clearance

    # connecting to the SITL
    vehicle = connect('udp:127.0.0.1:14550', wait_ready=True, baud=57600)

    control = ControlDrone(vehicle, hres=capture_width, vres=capture_height, hfov=hfov, vfov=vfov,
                           kp=kp, ki=ki, kd=kd, clearance_m=clearance_m)

    # initialize simulated video frames
    heli_path = (os.path.dirname(os.path.realpath(__file__))) + "/SimulationImage/heli.jpg"
    heli_sim = SimulateLandingPad(heli_path, vehicle)

    date_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    # for recording the video
    video_normal = cv2.VideoWriter("video_normal_{}.mp4".format(date_str), 0x7634706d, 10,
                                   (args.detection_width, args.detection_height))
    video_annotations = cv2.VideoWriter("video_annotations{}.mp4".format(date_str), 0x7634706d, 10,
                                        (args.detection_width, args.detection_height))

    control_df = [[date_str,
                   None, None,  # CX_P, CY_P,
                   None, None,  # SX_P, SY_P,
                   None, None,  # EX_P, EY_P,
                   None, None,  # OX_P, OY_P,
                   None, None,  # DLR_M, DFB_M,
                   None, None, None,  # self.kp, self.ki, self.kd,
                   None,  # Landing Target
                   None]]  # alt

    # setting the landing target
    # could be either 'heli' (indicating landing target is helipad)
    # 'empty' (indicating landing target is the nearest empty space)
    current_target = 'heli'

    heli_det_time = None  # variable to store the last time helipad was detected
    obs_unsafe_start_time = None  # variable to store the time when obstacles started becoming unsafe
    # Flag to store whether the current target is set to 'empty' due to the presence of obstacle for long period of time
    target_empty_due_to_obstacle = False
    OX_P = None  # Variable to store the control values for left/right movement of drone

    # operating the drone in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")

    # waiting for the mode to change to guided mode
    while True:
        if vehicle.mode == "GUIDED":
            break
        time.sleep(0.1)

    arm_and_takeoff(vehicle, 8)

    try:
        while True:
            if vehicle.location.global_relative_frame.alt < 1.5:
                print('Landing complete')
                vehicle.mode = VehicleMode("LAND")
                if len(control_df) > 1:
                    df = pd.DataFrame(control_df,
                                      columns=['Date', 'CX_P', 'CY_P', 'SX_P', 'SY_P', 'EX_P', 'EY_P', 'OX_P',
                                               'OY_P', 'DLR_M', 'DFB_M', 'kp', 'ki', 'kd', 'Landing Target',
                                               'Altitude'])
                    date_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                    df.to_csv('log_{}.csv'.format(date_str), index=False)
                heli_sim.release()
                cv2.destroyAllWindows()
                video_normal.release()
                video_annotations.release()
                break
            ret, rgb = heli_sim.read()
            if ret:
                rgb_track = rgb.copy()
                empty_frame_2D = get_empty_frames(rgb)
                # detection process
                orig_height, orig_width = rgb.shape[:2]
                detection_height = int((((detection_width / orig_width) * orig_height) // 32) * 32)
                img = cv2.resize(rgb, (detection_width, detection_height))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = np.moveaxis(img, -1, 0)
                img = torch.from_numpy(img).to(device)
                img = img.float() / 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                pred = yolo_model(img, augment=False)[0]
                pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None)

                helipad = None
                obstacles = None

                # For DeepSORT
                bboxes = []
                scores = []

                # coordinates of the centres of the frame
                CX_P = rgb.shape[1] / 2
                CY_P = rgb.shape[0] / 2

                if pred[0] is not None and len(pred[0]):
                    for n, p in enumerate(pred[0]):
                        score = np.round(p[4].cpu().detach().numpy(), 2)
                        label = Object_classes[int(p[5])]

                        if score > 0.5:
                            xmin = int(p[0] * rgb.shape[1] / detection_width)
                            ymin = int(p[1] * rgb.shape[0] / detection_height)
                            xmax = int(p[2] * rgb.shape[1] / detection_width)
                            ymax = int(p[3] * rgb.shape[0] / detection_height)

                            object_centre_x = (xmin + xmax) / 2
                            object_centre_y = (ymin + ymax) / 2

                            dist_centre = ((CX_P - object_centre_x) ** 2 +
                                           (CY_P - object_centre_y) ** 2) ** (1 / 2)

                            if label == 'helipad':
                                if (helipad is None) or (helipad is not None and dist_centre < helipad[0, 5]):
                                    helipad = np.array([[n, xmin, ymin, xmax, ymax, dist_centre, score, int(p[5])]])
                                    HXC_P = object_centre_x
                                    HYC_P = object_centre_y
                                    TL_X = xmin
                                    TL_Y = ymin
                                    HW = xmax - xmin
                                    HH = ymax - ymin

                                    # For DeepSORT
                                    box_heli = [int((xmin + xmax) / 2), int((ymin + ymax) / 2), abs(xmax - xmin), abs(ymax - ymin)]
                                    score_heli = score

                                elif (helipad is not None) and (dist_centre == helipad[0, 5]):
                                    if score > helipad[0, 6]:
                                        # choosing the helipad detection with the highest confidence score
                                        helipad = np.array([[n, xmin, ymin, xmax, ymax, dist_centre, score, int(p[5])]])
                                        HXC_P = object_centre_x
                                        HYC_P = object_centre_y
                                        TL_X = xmin
                                        TL_Y = ymin
                                        HW = xmax - xmin
                                        HH = ymax - ymin

                                        # For Deep SORT
                                        box_heli = [int((xmin + xmax) / 2), int((ymin + ymax) / 2), abs(xmax - xmin), abs(ymax - ymin)]
                                        score_heli = score

                            else:
                                empty_frame_2D = cv2.rectangle(empty_frame_2D, (xmin, ymin), (xmax, ymax), (0, 0, 0), -1)

                                if obstacles is None:
                                    obstacles = np.array([[xmin, ymin, xmax, ymax]])
                                else:
                                    obstacles = np.append(obstacles, np.array([[xmin, ymin, xmax, ymax]]), axis=0)

                                color = Object_colors[Object_classes.index(label)]
                                rgb_track = cv2.rectangle(rgb_track, (xmin, ymin), (xmax, ymax), color, 2)
                                rgb_track = cv2.putText(rgb_track, f'{label} ({str(score)})', (xmin, ymin),
                                                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 1, cv2.LINE_AA)
                if helipad is not None:
                    if target_empty_due_to_obstacle == False:
                        # current_target = 'heli'
                        heli_det_time = time.time()
                else:
                    if heli_det_time is None:
                        heli_det_time = time.time()
                    curr_time = time.time()
                    if curr_time - heli_det_time > 10:
                        if current_target != 'empty':
                            print('Helipad not detected for long time changing the landing target to an empty space')
                            current_target = 'empty'

                EMXC_P, EMYC_P, EM_R = DistanceTransform(empty_frame_2D)
                if current_target == 'heli':
                    if helipad is not None:
                        # Tracking using DeepSORT
                        bboxes.append(box_heli)
                        scores.append(score_heli)
                        outputs = deepsort.update(np.array(bboxes), scores, rgb)
                        if len(outputs) != 0:
                            bbox = outputs[:, :4]
                            identities = outputs[:, -1]
                            for i, box in enumerate(bbox):
                                x1, y1, x2, y2 = [int(i) for i in box]
                                id = int(identities[i]) if identities is not None else 0
                                if id == 1:

                                    color = [int((p * (id ** 2 - id + 1)) % 255) for p in palette]
                                    label_text = 'Helipad{}{:d}'.format("", id)
                                    t_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
                                    cv2.rectangle(rgb_track, (x1, y1), (x2, y2), color, 3)
                                    cv2.rectangle(rgb_track, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
                                    cv2.putText(rgb_track, label_text, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2,
                                                [255, 255, 255], 2)
                                    # Error values in pixel
                                    SX_P = ((x1 + x2) / 2)
                                    SY_P = ((y1 + y2) / 2)

                                    EX_P = SX_P - CX_P
                                    EY_P = -(SY_P - CY_P)  # opposite in case of forward backward

                                    OX_P, OY_P, DLR_M, DFB_M, kp, ki, kd, alt, obs_unsafe_start_time = \
                                        control.move_to_heli(EX_P=EX_P, EY_P=EY_P, HXMIN_P=x1, HYMIN_P=y1,
                                                             HXMAX_P=x2, HYMAX_P=y2, obstacles=obstacles,
                                                             obs_unsafe_time=obs_unsafe_start_time)

                                    curr_time = time.time()
                                    if obs_unsafe_start_time is not None:
                                        if curr_time - obs_unsafe_start_time > 20:
                                            current_target = 'empty'
                                            target_empty_due_to_obstacle = True
                                    print('Altitude:', alt)
                                    date_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                                    control_df.append(
                                        [date_str,
                                         CX_P, CY_P,
                                         SX_P, SY_P,
                                         EX_P, EY_P,
                                         OX_P, OY_P,
                                         DLR_M, DFB_M,
                                         kp, ki, kd,
                                         current_target,
                                         alt])

                else:
                    # Error values in pixel
                    EX_P = EMXC_P - CX_P
                    EY_P = -(EMYC_P - CY_P)  # opposite in case of forward backward
                    SX_P = EMXC_P
                    SY_P = EMYC_P
                    OX_P, OY_P, DLR_M, DFB_M, kp, ki, kd, alt = control.move_to_empty(EX_P, EY_P, EM_R)
                    date_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                    control_df.append([date_str,
                                       CX_P, CY_P,
                                       SX_P, SY_P,
                                       EX_P, EY_P,
                                       OX_P, OY_P,
                                       DLR_M, DFB_M,
                                       kp, ki, kd,
                                       current_target,
                                       alt])

                if OX_P is not None:
                    if OX_P > 0:
                        LR_P = 'RIGHT'
                    else:
                        LR_P = 'LEFT'

                    if OY_P > 0:
                        FB_P = 'FORWARD'
                    else:
                        FB_P = 'BACKWARD'

                    if DLR_M > 0:
                        LR_M = 'RIGHT'
                    else:
                        LR_M = 'LEFT'

                    if DFB_M > 0:
                        FB_M = 'FORWARD'
                    else:
                        FB_M = 'BACKWARD'

                    display_text1 = 'S(' + f"{SX_P:03.01f}" + ',' + f"{SY_P:03.01f}" + ') -C(' + f"{CX_P:03.01f}" + ',' + f"{CY_P:03.01f}" + ')='
                    display_text2 = 'E(' + f"{EX_P:03.01f}" + ',' + f"{EY_P:03.01f}" + ')>O(' + f"{OX_P:03.01f}" + ',' + f"{OY_P:03.01f}" + ')='
                    display_text3 = 'D(' + f"{DLR_M:03.01f}" + ',' + f"{DFB_M:03.01f}" + ')' + LR_P + ' ' + FB_P
                    display_text4 = 'Kp:' + f"{kp:03.02f}" + ', Ki:' + f"{ki:03.02f}" + ', Kd:' + f"{kd:03.02f}"

                    cv2.putText(rgb_track, display_text1, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    cv2.putText(rgb_track, display_text2, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    cv2.putText(rgb_track, display_text3, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)
                    cv2.putText(rgb_track, display_text4, (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    cv2.line(rgb_track, (int(SX_P), int(SY_P)), (int(CX_P), int(CY_P)), (255, 0, 0), 4)
                    cv2.line(rgb_track, (int(OX_P + CX_P), int(-OY_P + CY_P)), (int(CX_P), int(CY_P)), (0, 0, 255), 4)
                    control_df.append([date_str,
                                       CX_P, CY_P,
                                       SX_P, SY_P,
                                       EX_P, EY_P,
                                       OX_P, OY_P,
                                       DLR_M, DFB_M,
                                       kp, ki, kd,
                                       current_target,
                                       alt])

                else:
                    rgb_track = rgb

                cv2.imshow("CSI Camera", rgb)
                cv2.imshow("Annotated", rgb_track)
                video_normal.write(rgb)
                video_annotations.write(rgb_track)
                keyCode = cv2.waitKey(30)
                if keyCode == ord('q'):
                    if len(control_df) > 1:
                        df = pd.DataFrame(control_df,
                                          columns=['Date', 'CX_P', 'CY_P', 'SX_P', 'SY_P', 'EX_P', 'EY_P', 'OX_P',
                                                   'OY_P', 'DLR_M', 'DFB_M', 'kp', 'ki', 'kd', 'Landing Target',
                                                   'Altitude'])
                        date_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                        df.to_csv('log_{}.csv'.format(date_str), index=False)
                    heli_sim.release()
                    cv2.destroyAllWindows()
                    video_normal.release()
                    video_annotations.release()
                    break

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        if len(control_df) > 1:
            df = pd.DataFrame(control_df, columns=['Date', 'CX_P', 'CY_P', 'SX_P', 'SY_P', 'EX_P', 'EY_P', 'OX_P',
                                                   'OY_P', 'DLR_M', 'DFB_M','kp', 'ki', 'kd', 'Landing Target',
                                                   'Altitude'])
            date_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            df.to_csv('log_{}.csv'.format(date_str), index=False)
        heli_sim.release()
        cv2.destroyAllWindows()
        video_normal.release()
        video_annotations.release()


    finally:
        if len(control_df) > 1:
            df = pd.DataFrame(control_df, columns=['Date', 'CX_P', 'CY_P', 'SX_P', 'SY_P', 'EX_P', 'EY_P', 'OX_P',
                                                   'OY_P', 'DLR_M', 'DFB_M','kp', 'ki', 'kd', 'Landing Target',
                                                   'Altitude'])
            date_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            df.to_csv('log_{}.csv'.format(date_str), index=False)
        heli_sim.release()
        cv2.destroyAllWindows()
        video_normal.release()
        video_annotations.release()
