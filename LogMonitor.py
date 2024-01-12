import threading
from io import StringIO
import logging


class LogMonitor:
    """
    For monitoring the logs received from the Flight controller
    and confirm whether the mission is completed
    """
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.mission_started = threading.Event()
        self.mission_completed = threading.Event()

        self.str_io = StringIO()
        self.autopilot_logger = logging.getLogger('autopilot')
        self.autopilot_logger.setLevel(logging.DEBUG)
        self.streamHandler = logging.StreamHandler(self.str_io)
        self.autopilot_logger.addHandler(self.streamHandler)
        self.prev_logs_len = 0

        self.n_coms = 0

        self.end_the_thread = False
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        print('I am being printed from log capture')
        while True:
            try:
                if not self.end_the_thread:
                    logs = self.str_io.getvalue()
                    logs_list = logs.splitlines()

                    if self.prev_logs_len < len(logs_list):
                        if not self.mission_started.is_set():
                            print('Checking for flight plans')
                            if logs_list[-1] == 'Mission: 1 Takeoff':
                                cmds = self.vehicle.commands
                                cmds.download()
                                cmds.wait_ready()
                                self.n_coms = cmds.count
                                self.mission_started.set()

                        else:
                            if logs_list[-1] == 'Reached command #{}'.format(self.n_coms):
                                print('Mission Completed')
                                self.mission_completed.set()
                                break
                        self.prev_logs_len = len(logs_list)
                else:
                    break

            except Exception as e:
                print('Exception:', e)
                self.end_the_thread = True
                break

    def life_left(self):
        return self.t.is_alive()

    def mission_complete(self):
        return self.mission_completed.is_set()

    def release(self):
        self.end_the_thread = True
