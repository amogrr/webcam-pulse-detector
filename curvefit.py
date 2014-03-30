import datetime
import random
import matplotlib.pyplot as plt

from pylab import *
from numpy import *
from numpy.random import normal
from scipy.optimize import fmin

import numpy as np
from scipy.optimize import curve_fit
import numpy
import math
import scipy


from numpy import *
from pylab import *
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



i = 0
zero = 0
currx, curry = 0, 0
xo = []
yo = []
for line in open("MSPROJtest1/timedata - Copy.txt"):
	arr = line.split()
	if(len(arr)>0):
		currx = float(arr[0])
		curry = float(arr[1])

		if(i == 0):
			i+=1
			zero = float(arr[0])
		xo.append((currx-zero)*100000)
		yo.append(curry)

x1 = []
y1 = []
for line in open("MSPROJtest1/timedata - Copy (2).txt"):
    arr = line.split()
    if(len(arr)>0):
        currx = float(arr[0])
        curry = float(arr[1])

        if(i == 0):
            i+=1
            zero = float(arr[0])
        x1.append((currx-zero)*100000)
        y1.append(curry)
import random
print len(yo)
print len(y1)
newy1=[]
#print(zip(yo, y1))
for i in range (0,len(y1)):
    newy1.append(random.randint(0,200))
#y1 = newy1
newy1=[]
for i in range (0,len(yo)):
    newy1.append(len(yo)-i)
#yo = newy1
rmsdiff = 0
for (x, y) in zip(yo, y1):
    rmsdiff += (x - y) ** 2  # NOTE: overflow danger if the vectors are long!
rmsdiff = math.sqrt(rmsdiff / max(len(yo), len(y1)))
#print(rmsdiff/(max(max(xo),max(x1))-min(min(xo),min(x1))))
#import scipy.spatial
#print scipy.spatial.distance.cdist((xo,yo), (x1[:len(xo)],y1[:len(yo)]), 'euclidean')
from scipy.stats.stats import pearsonr
print(pearsonr(yo, y1[:len(yo)]))

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
    l=['original signal', 'signal with noise']
    l.extend(windows)

    legend(windows)
    
    
    #plot(x)
    
    w = 'hamming'
    
    #for w in windows:

    #t=linspace(xo[0],xo[-1],len(yo))
    #plot(t, yo,linestyle='--')
    t=linspace(xo[0],xo[-1],len(smooth(np.asarray(yo),10,w)))

    plot(t, smooth(np.asarray(yo),10,w))
    t=linspace(xo[0],xo[-1],len(smooth(np.asarray(y1[:len(yo)]),10,w)))
    global rmsdiff
    plot(t, smooth(np.asarray(y1[:len(yo)]),10,w))
    b = pearsonr(smooth(np.asarray(yo),10,w), smooth(np.asarray(y1[:len(yo)]),10,w))[0]
    a = str(math.trunc(b*100))  + "% correlation"
    text(3, 5, a, fontsize=30)
    #legend(l)
    title("")
    show()


if __name__=='__main__':
    smooth_demo()