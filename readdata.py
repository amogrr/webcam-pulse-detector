import datetime
import random
import matplotlib.pyplot as plt

from pylab import *
from numpy import *
from numpy.random import normal
from scipy.optimize import fmin

import numpy as np
from scipy.optimize import curve_fit

i = 0
zero = 0
currentx, currenty = 0, 0
xo = []
yo = []
for line in open("timedata.txt"):
	arr = line.split()
	if(len(arr)>0):
		currentx = float(arr[0])
		currenty = float(arr[1])

		if(i == 0):
			i+=1
			zero = float(arr[0])
		xo.append((currentx-zero)*100000)
		yo.append(currenty)
x1=[]
y1=[]
for line in open("heartdata.txt"):
	arr = line.split()
	if(len(arr)>0):
		currentx = float(arr[0])
		currenty = float(arr[1])/2

		if(i == 0):
			i+=1
			zero = float(arr[0])
		x1.append((currentx-zero)*100000)
		y1.append(currenty)
x2=[]
y2=[]
for line in open("smiledata.txt"):
	arr = line.split()
	if(len(arr)>0):
		currentx = float(arr[0])
		currenty = float(arr[1])/100

		if(i == 0):
			i+=1
			zero = float(arr[0])
		x2.append((currentx-zero)*100000)
		y2.append(currenty)
x3=[]
y3=[]
for line in open("eyedata.txt"):
	arr = line.split()
	if(len(arr)>0):
		currentx = float(arr[0])
		currenty = float(arr[1])/100

		if(i == 0):
			i+=1
			zero = float(arr[0])
		x3.append((currentx-zero)*100000)
		y3.append(currenty)
#print(x)
# plot
#plt.plot(x,y, 'g')

#plt.show()

import numpy

def smooth(x,window_len,window):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y




from numpy import *
from pylab import *

def smooth_demo():

    t=linspace(xo[0],xo[-1],len(xo))
    x=sin(t)
    xn=x+randn(len(t))*0.1
    y=smooth(x,11,"hanning")
    #print(xn)
    #print(str(x) + str(len(x)))
    #print(str(y) + str(len(y)))
    ws=31

   
     

    windows=[ 'hanning']
    l=['original affect curve', 'smoothed affect','heartdata','smiledata','eyedata']
    l.extend(windows)

    legend(windows)
    
    
    #plot(x)
    
    w = 'hamming'
    
    #for w in windows:

    t=linspace(xo[0],xo[-1],len(yo))
    plot(t, yo,linestyle=':')
    t=linspace(xo[0],xo[-1],len(smooth(np.asarray(yo),10,w)))

    #plot(t, smooth(np.asarray(yo),10,w))
    t=linspace(x1[0],x1[-1],len(y1))
    #plot(t, y1)
    t=linspace(x2[0],x2[-1],len(smooth(np.asarray(y2[20:]),10,w)))
    plot(t, smooth(np.asarray(y2[20:]),10,w))
    t=linspace(x3[0],x3[-1],len(y3))
    #plot(t, y3)
    legend(l)
    title("")
    show()


if __name__=='__main__':
    smooth_demo()
