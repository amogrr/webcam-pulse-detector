import cv2, time
import urllib2, base64
import numpy as np
    
class Camera(object):

    def __init__(self, camera = 0):
        
        print("initializing camera")
        self.cam = cv2.VideoCapture(camera)
        time.sleep(3)
        print("Camera ready!")
        if not self.cam:
            raise Exception("Camera not accessible")

        self.shape = self.get_frame().shape
        print(self.shape)
    def get_frame(self):
        _,frame = self.cam.read()
        return frame

    def release(self):
        self.cam.release()
