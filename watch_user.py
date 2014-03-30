from lib.device import Camera
from lib.processors import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey,destroyWindow, moveWindow
import numpy as np      
import datetime
###########################
import cv2.cv as cv
import time
import Image
import threading
###########################
import matplotlib
import datetime
import matplotlib.dates as md
import time
import sys
import math
##########################
#global total
#total = 0
#flag = 0
##########################
#Default heart rate value, can be changed later
dhr = 50
eyetot = 0
norm = 0
##########################
import os.path
if (os.path.isfile('diagnostics.txt')) == False:
    print("Diagnostics file not found. Run diagnostics first!")
    sys.exit()
else:
    diag = open('diagnostics.txt','r')
    z = 0
    for line in diag:
        if z == 0:
            z+=1
            dhr = float(line.rstrip('\n'))
        else:
            norm = float(line.rstrip('\n'))

f = open('timedata.txt','w')
e = open('eyedata.txt','w')
hr = open('heartdata.txt','w')
sm = open('smiledata.txt','w')
cv.NamedWindow("camera", 1)
capture = cv.CreateCameraCapture(0)

#font = cv.CvFont
font = cv.InitFont(1, 1, 1, 1, 1, 1)

width = None
height = None
width = 320
height = 240
smileneighbour = 0
smilecount = 0

if width is None:
    width = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_WIDTH))
else:
    cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_WIDTH,width)    

if height is None:
    height = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_HEIGHT))
else:
    cv.SetCaptureProperty(capture,cv.CV_CAP_PROP_FRAME_HEIGHT,height) 

result = cv.CreateImage((width,height),cv.IPL_DEPTH_8U,3) 

mqLoop = 0



#openCV functions
def Load():

    return (faceCascade, smileCascade)

def Display(image):
    cv.NamedWindow("Smile Test")
    cv.ShowImage("Smile Test", image)
    cv.WaitKey(0)
    cv.DestroyWindow("Smile Test")

def DetectRedEyes(image, faceCascade, smileCascade, eyeCascade):
    min_size = (20,20)
    image_scale = 2
    haar_scale = 1.1
    min_neighbors = 2
    haar_flags = 0

    # Allocate the temporary images
    gray = cv.CreateImage((image.width, image.height), 8, 1)
    smallImage = cv.CreateImage((cv.Round(image.width / image_scale),cv.Round (image.height / image_scale)), 8 ,1)

    # Convert color input image to grayscale
    cv.CvtColor(image, gray, cv.CV_BGR2GRAY)

    # Scale input image for faster processing
    cv.Resize(gray, smallImage, cv.CV_INTER_LINEAR)

    # Equalize the histogram
    cv.EqualizeHist(smallImage, smallImage)

    # Detect the faces
    faces = cv.HaarDetectObjects(smallImage, faceCascade, cv.CreateMemStorage(0),
    haar_scale, min_neighbors, haar_flags, min_size)
    global norm
    # If faces are found
    if faces:
        
        #print faces
        ratio = 1.
        for ((x, y, w, h), n) in faces:
        # the input to cv.HaarDetectObjects was resized, so scale the
        # bounding box of each face and convert it to two CvPoints
            #print "face"
            if h!=0:
                ratio = h/norm

            pt1 = (int(x * image_scale), int(y * image_scale))
            pt2 = (int((x + w) * image_scale), int((y + h) * image_scale))
            # print pt1
            # print pt2
            #cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 1, 8, 0)
            #cv.PutText(image, "face"+str(h), pt1, font, cv.RGB(255, 0, 0))
            face_region = cv.GetSubRect(image,(x,int(y + (h/4)),w,int(h/2)))

            #split face
            #cv.Rectangle(image, (pt1[0],(pt1[1] + (abs(pt1[1]-pt2[1]) / 2 ))), pt2, cv.RGB(0,255,0), 1, 8, 0)
            #cv.PutText(image, "lower", (pt1[0],(pt1[1] + (abs(pt1[1]-pt2[1]) / 2 ))), font, cv.RGB(0, 255, 0))
            cv.SetImageROI(image, (pt1[0],
                               (pt1[1] + int(abs(pt1[1]-pt2[1]) * 0.625 )),
                               pt2[0] - pt1[0],
                               int((pt2[1] - (pt1[1] + int(abs(pt1[1]-pt2[1]) * 0.625 ))))))
            
            smiles = cv.HaarDetectObjects(image, smileCascade, cv.CreateMemStorage(0), 1.1, 5, 0, (15,15))
        
            if smiles:
                #print smiles          
                for smile in smiles:
                    cv.Rectangle(image,
                    (smile[0][0],smile[0][1]),
                    (smile[0][0] + smile[0][2], smile[0][1] + smile[0][3]),
                    cv.RGB(0, 0, 255), 1, 8, 0)
                    sizer = (smile[0][2]/ratio+smile[0][3]/ratio)#+(smile[1]/ratio))
                    #sizer = math.trunc(sizer)
                    #cv.PutText(image, "smile", (smile[0][0],smile[0][1]), font, cv.RGB(0, 0, 255))

                    cv.PutText(image,str(math.trunc(sizer**2)), (smile[0][0], smile[0][1] + smile[0][3] + 10), font, cv.RGB(0, 0, 255))
                    #print ((abs(smile[0][1] - smile[0][2]) / abs(pt1[0] - pt2[0])) * 100) 
                    
                    global smileneighbour 
                    smileneighbour = sizer**2*2
            cv.ResetImageROI(image)
            #############################################################################
            #############################################################################
            cv.SetImageROI(image, (pt1[0], pt1[1], int(pt2[0]-pt1[0]), int(pt2[1] - pt1[1])) )
            eyes = cv.HaarDetectObjects(image, eyeCascade,cv.CreateMemStorage(0),haar_scale, 5,haar_flags, (15,15))
            if eyes:
                # For each eye found
                iii = 0
                #print eyes
                for eye in eyes:
                    # Draw a rectangle around the eye
                   cv.Rectangle(image,(eye[0][0],eye[0][1]),(eye[0][0] + eye[0][2],eye[0][1] + eye[0][3]), cv.RGB(0, 0, 255), 1, 8, 0)
                   a = math.trunc(float(eye[1])/ratio)
                   cv.PutText(image,str(a), (eye[0][0], eye[0][1] + eye[0][3]), font, cv.RGB(0, 0, 255))
                   global eyetot
                   eyetot += float(eye[1]*eye[1])/ratio
                   iii+=1
                   if iii==2:
                       iii = 0
                       break
            cv.ResetImageROI(image)
    cv.ResetImageROI(image)
    return image

faceCascade = cv.Load("haarcascade_frontalface_alt.xml")
smileCascade = cv.Load("smileD\smiled_01.xml")#"haarcascade_smile.xml")
eyeCascade = cv.Load("haarcascade_eye.xml")


##########################
class AnalyseEmotion(object):
    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered 
    over time, and the detected person's pulse is estimated.
    """
    def __init__(self):
        #Imaging device - must be a connected camera (not an ip camera or mjpeg
        #stream)
        self.camera = Camera(camera=0) #first camera by default
        
        self.w,self.h = 0,0
        self.pressed = 0
        #Containerized analysis of recieved image frames (an openMDAO assembly)
        #is defined next.

        #This assembly is designed to handle all image & signal analysis,
        #such as face detection, forehead isolation, time series collection,
        #heart-beat detection, etc. 

        #Basically, everything that isn't communication
        #to the camera device or part of the GUI
        self.processor = findFaceGetPulse(bpm_limits = [50,160],
                                          data_spike_limit = 2500.,
                                          face_detector_smoothness = 10.)  

        #Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Cardiac info - raw signal, filtered signal, and PSD"

        #Maps keystrokes to specified methods
        #(A GUI window must have focus for these to work)
        self.key_controls = {"s" : self.toggle_search,
                             "d" : self.toggle_display_plot,
                             "f" : self.write_csv}
        
    def write_csv(self):
        """
        Writes current data to a csv file
        """
        bpm = " " + str(int(self.processor.measure_heart.bpm))
        fn = str(datetime.datetime.now()).split(".")[0] + bpm + " BPM.csv"
        
        data = np.array([self.processor.fft.times, 
                         self.processor.fft.samples]).T
        np.savetxt(fn, data, delimiter=',')
        


    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated. 
        """
        state = self.processor.find_faces.toggle()
        if not state:
        	self.processor.fft.reset()
        print "face detection lock =",not state

    def toggle_display_plot(self):
        """
        Toggles the data display.
        """
        if self.bpm_plot:
            print "bpm plot disabled"
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print "bpm plot enabled"
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w,0)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        """
        plotXY([[self.processor.fft.times, 
                 self.processor.fft.samples],
                [self.processor.fft.even_times[4:-4], 
                 self.processor.measure_heart.filtered[4:-4]],
                [self.processor.measure_heart.freqs, 
                 self.processor.measure_heart.fft]], 
               labels = [False, False, True],
               showmax = [False,False, "bpm"], 
               label_ndigits = [0,0,0],
               showmax_digits = [0,0,1],
               skip = [3,3,4],
               name = self.plot_title, 
               bg = self.processor.grab_faces.slices[0])

    def key_handler(self):    
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        """

        self.pressed = waitKey(10) & 255 #wait for keypress for 10 ms
        if self.pressed == 27: #exit program on 'esc'
            print "exiting..."
            self.camera.cam.release()
            exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        """
        Single iteration of the application's main loop.
        """
        # Get current image frame from the camera
        frame = self.camera.get_frame()
        self.h,self.w,_c = frame.shape
        

        #display unaltered frame
        #imshow("Original",frame)

        #set current image frame to the processor's input
        self.processor.frame_in = frame
        #process the image frame to perform all needed analysis
        self.processor.run()
        #collect the output frame for display
        output_frame = self.processor.frame_out

        #show the processed/annotated output frame
        imshow("Processed",output_frame)

        #create and/or update the raw data display if needed
        global smileneighbour, mqLoop, smilecount, eyetot
        #if self.bpm_plot:
            #self.make_bpm_plot()  
        if mqLoop >= 1:
            x = str(datetime.datetime.now())
            sm.write(str(md.datestr2num(x)) + " " + str(smileneighbour) + "\n")
            e.write(str(md.datestr2num(x)) + " " + str(eyetot) + "\n")
            hr.write(str(md.datestr2num(x)) + " " + str(self.processor.show_bpm_text.bpm) + "\n")
            smileneighbour+= 2*eyetot
            smileneighbour/=100
            if (self.processor.show_bpm_text.bpm) > dhr:
                #print (self.processor.fft.samples[-1]/2, self.processor.fft.samples[-1]-dhr/2)
                #overbeat = (self.processor.fft.samples[-1]-dhr)*(self.processor.fft.samples[-1]-dhr)
                smileneighbour += (self.processor.show_bpm_text.bpm-dhr)
            
            
            f.write(str(md.datestr2num(x)) + " " + str(smileneighbour) + "\n")
            mqLoop = 0
        else:
            mqLoop+= 0.9    
        img = cv.QueryFrame(capture)    
        smileneighbour = 0
        eyetot = 0
        if img:
            image = DetectRedEyes(img, faceCascade, smileCascade, eyeCascade)
            cv.ShowImage("camera", image)
        #handle any key presses
        self.key_handler()

if __name__ == "__main__":
    App = AnalyseEmotion()
    while True:
        App.main_loop()
