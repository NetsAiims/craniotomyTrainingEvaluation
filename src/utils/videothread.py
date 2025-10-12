from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import cv2
import platform
import datetime
import math
import ffmpegcv

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_timestamp_signal = pyqtSignal(str)

    def __init__(self, capture):
        super().__init__()
        self._run_flag = True
        self._recording = False
        self._pause_recording = False
        self.capture = capture
        self.frames_count = 0
        self.videowriter = None

    def run(self):
        try:
            while self._run_flag:
                ret, cv_img = self.capture.read()
                if ret:
                    self.change_pixmap_signal.emit(cv_img)
                if self._recording and not self._pause_recording:
                    self.videowriter.write(cv_img)
                    self.frames_count += 1
                    self.change_timestamp_signal.emit(str(datetime.timedelta(seconds=int(self.frames_count / self.video_fps))))
                    if not self._recording:
                        self.videowriter.release()
                        self.videowriter = None
                elif self._recording and self._pause_recording:
                    self.change_timestamp_signal.emit(str(datetime.timedelta(seconds=int(self.frames_count / self.video_fps))))
                elif not self._recording:
                    self.change_timestamp_signal.emit("NA")
                    if self.videowriter is not None:
                        self.videowriter.release()
                        self.videowriter = None
        except:
            if self.capture is None:
                print("Please select capture device")
            elif self.videowriter is not None:
                self.videowriter.release()
                self.videowriter = None

        # shut down capture system
    
    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
    
    def pause_recording(self):
        self._pause_recording = True

    def restart_recording(self):
        self._pause_recording = False
    
    def stop_recording(self):
        self._recording = False
        self._pause_recording = True
        self.frames_count = 0

    def is_recording(self):
        return self._recording
    
    def is_streaming(self):
        return self._run_flag
    
    def start_recording(self, file_name):
        frame_width = int(self.capture.get(3))
        frame_height = int(self.capture.get(4))
        if math.isnan(self.capture.get(cv2.CAP_PROP_FPS)):
            self.video_fps = 25.0
        else:
            self.video_fps = self.capture.get(cv2.CAP_PROP_FPS)

        self.videowriter = ffmpegcv.VideoWriter(f"{file_name}.avi", None, self.video_fps)
        if self.videowriter is None:
            return
        self._recording = True
        self._pause_recording = False
