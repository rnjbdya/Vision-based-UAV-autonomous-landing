import cv2, queue, threading
import numpy as np


class VideoCapture:
    def __init__(self, capture_width=640, capture_height=480, detection_width=224, detection_height=224, framerate=30,
                 flip_method=0, source='rs'):
        self.source = source

        # the picam and the intel real sense cam have different initializations
        if self.source == 'pi':
            gstream = self.gstreamer_pipeline(capture_width, capture_height, detection_width, detection_height, framerate,
                                              flip_method)
            self.cap = cv2.VideoCapture(gstream, cv2.CAP_GSTREAMER)

        elif self.source == 'rs':
            import pyrealsense2 as rs
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
            self.device = self.pipeline_profile.get_device()
            self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
            self.found_rgb = False
            for s in self.device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    self.found_rgb = True
                    break
            if not self.found_rgb:
                print("The source has been selected as intel realsense camera "
                      "but a depth camera with color sensor not detected")
                exit(0)
            self.config.enable_stream(rs.stream.depth, capture_width, capture_height, rs.format.z16, framerate)

            if self.device_product_line == 'L500':
                self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
            else:
                self.config.enable_stream(rs.stream.color, capture_width, capture_height, rs.format.bgr8, 30)

            # Start streaming
            self.pipeline.start(self.config)

        self.q_frames = queue.Queue(maxsize=4)
        self.q_depth = queue.Queue(maxsize=4)
        self.end_the_thread = False
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    # defining the gstreamer pipeline for the picam
    def gstreamer_pipeline(self, capture_width=640, capture_height=480, detection_width=224, detection_height=224,
                           framerate=30, flip_method=0):
        return ("nvarguscamerasrc ! "
                "video/x-raw(memory:NVMM), "
                "width=(int)%d, height=(int)%d, "
                "format=(string)NV12, framerate=(fraction)%d/1 ! "
                "nvvidconv flip-method=%d ! "
                "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
                "videoconvert ! "
                "video/x-raw, format=(string)BGR ! appsink"
                % (
                    capture_width,
                    capture_height,
                    framerate,
                    flip_method,
                    detection_width,
                    detection_height,
                )
                )

    # read frames as soon as they are available and keep only the most recent
    def _reader(self):
        if self.source == 'pi':
            print('I am being printed from the video capture thread.')
            while True:
                try:
                    if not self.end_the_thread:
                        ret, frame = self.cap.read()
                        if not ret:
                            break
                        if not self.q_frames.empty():
                            try:
                                self.q_frames.get_nowait()  # discard previous (unprocessed) frame
                            except queue.Empty:
                                pass
                        self.q_frames.put((ret, frame))

                        if not self.q_depth.empty():
                            try:
                                self.q_depth.get_nowait()  # discard previous (unprocessed) frame
                            except queue.Empty:
                                pass
                        self.q_depth.put(frame)
                    else:
                        self.cap.release()
                        break
                except:
                    self.cap.release()
                    self.end_the_thread = True

        elif self.source == 'rs':
            while True:
                try:
                    if not self.end_the_thread:
                        # Wait for a coherent pair of frames: depth and color
                        frames = self.pipeline.wait_for_frames()
                        depth_frame = frames.get_depth_frame()
                        color_frame = frames.get_color_frame()
                        if not depth_frame or not color_frame:
                            continue

                        # Convert images to numpy arrays
                        depth_image = np.asanyarray(depth_frame.get_data())
                        color_image = np.asanyarray(color_frame.get_data())

                        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03),
                                                           cv2.COLORMAP_JET)
                        depth_colormap_dim = depth_colormap.shape
                        color_colormap_dim = color_image.shape

                        if depth_colormap_dim != color_colormap_dim:
                            color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                     interpolation=cv2.INTER_AREA)

                        if not self.q_frames.empty():
                            try:
                                self.q_frames.get_nowait()  # discard previous (unprocessed) frame
                            except queue.Empty:
                                pass
                        self.q_frames.put((True, color_image))

                        if not self.q_depth.empty():
                            try:
                                self.q_depth.get_nowait()  # discard previous (unprocessed) frame
                            except queue.Empty:
                                pass
                        self.q_depth.put(depth_image)

                    else:
                        try:
                            self.pipeline.stop()
                            break
                        except:
                            break

                except Exception as e:
                    print('Exception during video processing', e)
                    self.end_the_thread = True

    def read_frame(self):
        return self.q_frames.get()

    def read_depth(self):
        return self.q_depth.get()

    def release(self):
        print('Stopping Video processing')
        self.end_the_thread = True
