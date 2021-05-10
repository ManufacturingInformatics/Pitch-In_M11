from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import color
import os
import cv2
import csv
from itertools import zip_longest
from scipy.stats import gaussian_kde
import h5py

# get current working directory
curr_cwd = os.getcwd()

def readH264(path,flag='mask'):
   """ Read in image data from target H264 video file as float16
	
	 path : target file path
	 flag : how to process the data, default='mask' means the data is masked
   """
   # known size of the images
   rows = 128
   cols = 128

   # read in raw bytes as a 1D array
   arr = np.fromfile(path,dtype='uint16')

   if flag=='mask':
      ## update values based on code
      # get code
      code_array = np.bitwise_and(arr,0xF000)
      # CODE_VAL_SEUIL2
      arr[code_array==0xD000] = 0xF800
      # CODE_VAL_CONTOUR
      arr[code_array==0xB000] = 0xF81F
      # CODE_VAL_MAX
      arr[code_array==0xC000] = 0x0000
      # CODE_VAL_SEUIL1
      arr[code_array==0xE000] = 0x001F

   ## just lower 12-bits
   arr = np.bitwise_and(arr,0x0FFF)

   ## convert data to frames
   # break the data into chunks that are 1d frames
   frames_set = np.split(arr,int(arr.shape[0]/(rows*cols)))
   # combined frames together into a 3d array and interpret as float16 data type
   return np.dstack([np.reshape(f,(rows,cols)) for f in frames_set]).astype('float16')

def readXMLData(path):
    import xml.etree.ElementTree as ET
    """ Reads data from the given laser machine XML file

        =====ONLY READS FIRST DATA FRAME=====
        Typically only one frame in the file anyway

        path : file path to xml file

        Returns information as a list in the order
        header,time,torque,vel,x,y,z

        header : header information from the Data Frame in XML file
            time   : Sampling time vector for the values
            torque : Current applied to the motor to generate torque (A)
            vel    : motor velocity (mm/s)
            x      : motor position along x-axis (mm)
            y      : motor position along y-axis (mm)
            z      : motor position along z=axis (mm)
            
        time,torque,vel,x,y,z are lists of data

        header is a dictionary with the following entries:
            - "Date-Time" : Datetime stamp of the first Data Frame
            
            - "data-signal-0" : Dictionary of information about torque
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-1" : Dictionary of information about velocity
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-2" : Dictionary of information about x
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-3" : Dictionary of information about y
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
                
            - "data-signal-4" : Dictionary of information about z
                with the following items:
                + "interval" : time between samples
                + "description" : description of the signal
                + "count" : number of data points
                + "unitType" : description of what the data represents
    """
    # parse xml file to a tree structure
    tree = ET.parse(path)
    # get the root/ beginning of the tree
    #root  = tree.getroot()
    # get all the trace data
    log = tree.findall("traceData")
    # get the number of data frames/sets of recordings
    log_length = len(log)

    # data sets to record
    x = []
    y = []
    z = []
    torque = []
    vel = []
    time = []
    header = {}
    
    ## read in log data
    # for each traceData in log
    for f in log:
        # only getting first data frame as it is known that there is only one frame
        # get the header data for data frame
        head = f[0].findall("frameHeader")

        # get date-timestamp for file
        ### SYNTAX NEEDS TO BE MODIFIED ###
        temp = head[0].find("startTime")
        if temp == None:
            header["Date-Time"] = "Unknown"
        else:
            header["Date-Time"] = temp
            
        # get information about data signals
        head = f[0].findall("dataSignal")
        # iterate through each one
        for hi,h in enumerate(head):
            # create entry for each data signal recorded
            # order arranged to line up with non-time data signals
            header["data-signal-{:d}".format(hi)] = {"interval": h.get("interval"),
                                                     "description": h.get("description"),
                                                     "count":h.get("datapointCount"),
                                                     "unitType":h.get("unitsType")}
            # update unitType to something meaningful                            
            if header["data-signal-{:d}".format(hi)]["unitType"] == 'posn':
                header["data-signal-{:d}".format(hi)]["unitType"] = 'millimetres'
            elif header["data-signal-{:d}".format(hi)]["unitType"] == 'velo':
                header["data-signal-{:d}".format(hi)]["unitType"] = 'mm/s'
            elif header["data-signal-{:d}".format(hi)]["unitType"] == 'current':
                header["data-signal-{:d}".format(hi)]["unitType"] = 'Amps'
                                
        # get all recordings in data frame
        rec = f[0].findall("rec")
        # parse data and separate it into respective lists
        for ri,r in enumerate(rec):
            # get timestamp value
            time.append(float(r.get("time")))
            
            # get torque value
            f1 = r.get("f1")
            # if there is no torque value, then it hasn't changed
            if f1==None:
                # if there is a previous value, use it
                if ri>0:
                    torque.append(float(torque[ri-1]))
            else:
                torque.append(float(f1))

            # get vel value
            f2 = r.get("f2")
            # if there is no torque value, then it hasn't changed
            if f2==None:
                # if there is a previous value, use it
                if ri>0:
                    vel.append(float(vel[ri-1]))
            else:
                vel.append(float(f2))

            # get pos1 value
            val = r.get("f3")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    x.append(float(x[ri-1]))
            else:
                x.append(float(val))

            # get pos2 value
            val = r.get("f4")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    y.append(float(y[ri-1]))
            else:
                y.append(float(val))

            # get pos3 value
            val = r.get("f5")
            # if there is no torque value, then it hasn't changed
            if val==None:
                # if there is a previous value, use it
                if ri>0:
                    z.append(float(z[ri-1]))
            else:
                z.append(float(val))

    return header,time,torque,vel,x,y,z


# functions for converting temperature between celcius and kelvin
def CelciusToK(C):
    return float(C)+273.15

def KelvinToC(K):
    return float(K)-273.15

# starter temperature, room temperature
T0 = CelciusToK(23.0)

# melting temperature of 316L
Tm = 1390 # C
Tm_K = CelciusToK(Tm) # K

### ALL TEMPS IN C
#temperature resolution for [0-20] C or [0 - 293.15] K 
starter_res = 200

## temperature conductivity
print("Creating thermal conductivity data")
def genKdata():
    # temp range [0-20] avg
    K_range_1 = np.full(starter_res,16.3)
    # temp range [20-100]
    K_range_2 = np.full(int(starter_res*(80/20)),(16.3+14.6)/2)
    # temp range [100-500]
    K_range_3 = np.full(int(starter_res*(400/20)),21.5)
    # temp data, complete
    K_data = np.concatenate([K_range_1,K_range_2,K_range_3],axis=0)
    # temperature range for plotting data
    K_temp_range = np.linspace(0.0,CelciusToK(500.0),K_data.shape[0])
    return K_data,K_temp_range

res_scale = 2
K_data,K_temp_range = genKdata()
# temperature range to test fitting on
T_test_data = np.linspace(0,CelciusToK(1500),res_scale*(1500/20))


### interpolate data
## thermal conductivity
# returns a function that can be used later
# s parameter is the min square distance between interpolation and data
K_spline = UnivariateSpline(K_temp_range,K_data)
K_ispline = InterpolatedUnivariateSpline(K_temp_range,K_data)

def Kp(Ks,T,Tm,scale=0.01):
    """ Thermal conductivity of powder
        
        Ks : Thermal condutivity of solid material, numpy array
        T  : Temperature data to genereate Ks, K
        Tm : Melting point of material, K

        Taken to be fraction of solid thermal conductivity when
        temperature is less than the melting point, Tm.

        Source : 3-Dimensional heat transfer modeling for laser 
        powder-bed fusion additive manufacturing with volumetric 
        heat sources based on varied thermal conductivity and 
        absorptivity (2019), Z. Zhang et. al.

        Returns thermal conductivity for powder as approximated
    """

    from numpy import where,asarray,copy
    # make a copy of solid thermal conductivity matrix
    Kp = np.copy(Ks)
    # for all values where the temperature is less than the melting point
    # scale the thermal conductivity values
    Kp[where(T<Tm)] *= scale
    # return modified matrix
    return Kp

Kp_data = Kp(K_data,K_temp_range,Tm_K)
Kp_spline = UnivariateSpline(K_temp_range,Kp_data)
Kp_ispline = InterpolatedUnivariateSpline(K_temp_range,Kp_data)

## specific heat conductivity
print("Creating specific heat capacity data")
def genCdata():
    # temp range [0-20]
    C_range_1 = np.full(starter_res,450.0)
    # temp range [20-93]
    C_range_2 = np.full(int(starter_res*(73/20)),(485.0+500.0)/2.0)
    # temp range [93-100]
    C_range_3 = np.full(int(starter_res*(7/20)),500.0)
    # specific heat capacity, complete
    C_data = np.concatenate([C_range_1,C_range_2,C_range_3],axis=0)
    # temperature data for plotting data
    C_temp_range = np.linspace(0,CelciusToK(100),C_data.shape[0])
    return C_data,C_temp_range

C_data,C_temp_range = genCdata()

C_spline = UnivariateSpline(C_temp_range,C_data)
C_ispline = InterpolatedUnivariateSpline(C_temp_range,C_data)

# volume expansion factor, mm^3/mm^3.C
# same as m^3/m^3.K
# as both C and K are absolute units of measurement, a change is C is the same in K
# converting between mm to m cancels each other out
print("Generating volume expansion factor data")
def genEvData():
    """ Generate data for volume expansion factor based off linear expansion factors """
    # for T = [20-100]
    Ev_range_1 = np.full(int(starter_res*(80/20)),3.0*(16.6+18.2+19.4)/3.0)
    # for T = [100-500]
    Ev_range_2 = np.full(int(starter_res*(400/20)),3.0*(18.2+19.4)/2.0)
    # for T = [500-1000]
    Ev_range_3 = np.full(int(starter_res*(500/20)),3.0*19.4)
    # combine data together
    Ev_data = np.concatenate([Ev_range_1,Ev_range_2,Ev_range_3],axis=0)
    # temperature data for temperature range so it can be plotted
    Ev_temp_range = np.linspace(CelciusToK(20.0),CelciusToK(1000.0),Ev_data.shape[0])
    # return data and temp data
    return Ev_data,Ev_temp_range

Ev_data, Ev_temp_range = genEvData()

Ev_spline = UnivariateSpline(Ev_temp_range,Ev_data)
Ev_ispline = InterpolatedUnivariateSpline(Ev_temp_range,Ev_data)

def genDensityData(Ev):
    """ Generate density data based of thermal volumetric expansion 
        
        Ev : spline for thermal volumetric expansion
    """
    # for dT = [20 - 100] C
    p_range_1 = np.full(int(starter_res*(80/20)),(7900.0+8000.0+8027.0)/3.0)
    ## calculate density data for dT =[100 - 1000] C using Ev spline and prev range value as starter point
    p_range_2 = []
    for dT in np.linspace(0.0,900.0,int(starter_res*(20/900))):
        p_range_2.append(p_range_1[0]/(1+Ev(dT)*dT))
    # convert to array
    p_range_2 = np.array(p_range_2)
    # combine data ranges together
    p_data = np.concatenate([p_range_1,p_range_2],axis=0)
    # create temperature data
    p_temp_range = np.linspace(CelciusToK(20.0),CelciusToK(1000.0),p_data.shape[0])
    return p_data, p_temp_range

p_data,p_temp_range = genDensityData(Ev_ispline)

p_spline = UnivariateSpline(p_temp_range,p_data)
p_ispline = InterpolatedUnivariateSpline(p_temp_range,p_data)

def genThermalDiff(K,p,C,T):
    """ Generate thermal diffusivity data for solid material using previous splines

        K : thermal conductivity spline for solid material
        p : solid density spline
        C : specific heat capacity spline
        T : temperature data matrix

        returns thermal diffusivity data for the temperature range
    """

    return K(T)/(p(T)*C(T))

def genThermalDiff_powder(K,p,C,T):
    """ Generate thermal diffusivity data for powder using previous splines

        K : thermal conductivity spline for powder material
        p : solid density spline
        C : specific heat capacity spline
        T : temperature data matrix
        Tm : metling temperature of the solid material

        returns thermal diffusivity data for the temperature range
    """
    # generate thermal conductivity data
    # then modifiy it according to approximation function
    return Kp(K(T),T,Tm)/(p(T)*C(T))

# thermal diffusivity of the solid material
Ds_data = genThermalDiff(K_ispline,p_ispline,C_ispline,T_test_data)

Ds_spline = UnivariateSpline(T_test_data,Ds_data)
Ds_ispline = InterpolatedUnivariateSpline(T_test_data,Ds_data)

# thermal diffusivity using thermal conductivity scaling approximation
Dp_data = genThermalDiff_powder(Kp_ispline,p_ispline,C_ispline,T_test_data)

Dp_spline = UnivariateSpline(T_test_data,Dp_data)
Dp_ispline = InterpolatedUnivariateSpline(T_test_data,Dp_data)

## emissivity
# emissivity of stainless steel
e_ss = 0.53
# emissivity of steel galvanised
e_p = 0.28

# function for converting radiative heat to temperature
def Qr2Temp(qr,e,T0):
    """ Function for converting radiative heat to surface temperature

        qr : radiative heat W/m2
        e  : emissivity of the material [0.0 - 1.0]
        T0 : starting temperature, K

        Returns numpy array surface temperature in Kelvin using Stefan-Boltzman theory

        qr = e*sigma*(T^4 - T0^4)
        => T = ((qr/(e*sigma))**4.0 + T0**4.0)**0.25
    """
    from scipy.constants import Stefan_Boltzmann as sigma
    from numpy import asarray, nan_to_num,full
    # if emissivity is 0.0, return T0
    # result of setting qr/(e*sigma)
    # avoids attempt to divide by 0
    if e==0.0:
        #print("emissivity is 0")
        # if T0 is a numpy array and then return T0 as is
        if type(T0)==numpy.ndarray:
            return T0
        # if T0 is a single value float
        elif type(T0)==float:
            # construct and return a matrix of that value
            return full(qr.shape,T0,qr.dtype)
    else:
        # calculate left hand portion of addition
        # due to e_ss*sigma being so small (~10^-8)
        div = (e*sigma)**-1.0
        #qr/(e*sigma)
        a = (qr*div)
        # T0^4
        #print("T0:",T0)
        b = asarray(T0,dtype=np.float32)**4.0
        #print("To^4:",b)
        # (qr/(e*sigma)) + T0^4
        c = a+b
        return (c)**0.25

def Temp2IApprox(T,T0,K,D,t):
    """ Function for converting temperature to power density

        K : thermal conductivity, spline fn
        D : thermal diffusivity, spline fn
        t : interaction time between T0 and T
        T : temperature K, numpy matrix
        T0: starter temperature K , constant

        Returns power density approximation ndarray and laser power estimate

        Approximation based on the assumption of uniform energy for the given area
    """
    # get numpy fn to interpret temperature as matrix and get pi constant
    from numpy import asarray, pi, sqrt, abs
    # temperature difference
    Tdiff = (T-T0)
    # thermal conductivity matrix
    K = K(Tdiff)
    # thermal diffusivity matrix
    D = D(Tdiff)
    # 2*sqrt(Dt/pi)
    a = ((D*t)/np.pi)
    # result of sqrt can be +/-
    # power density cannot be negative 
    b = (2.0*np.sqrt(a))
    temp = K*Tdiff
    # K*(T-T0)/(2*sqrt(Dt/pi))
    return abs(temp/b)

def powerEstHoughCircle(circle,I,PP):
   ''' Estimating power using circle found in power density matrix by Hough Circle algorithm

       circle : Portion of the Hough Circle results matrix representing one circle. It's an array
                of a list of three values [x,y,radius]
       I : Power density matrix, W/m2
       PP : Pixel Pitch, m

       This is designed to be used with Numpy's apply along matrix command as applied to results

       Return sums the values within circle and multiplies by the area
   '''
   # create empty mask to draw results on 
   mask = np.zeros(I.shape[:2],dtype='uint8')
   # draw filled circle using given parameters   
   cv2.circle(mask,(*circle[:2],),circle[2],(255),cv2.FILLED)
   # find where it was drawn
   i,j = np.where(mask==255)
   # sum the power density values in that area and multiply by area
   return np.sum(I[i,j])*(np.pi*(circle[2]*PP)**2.0)

def diffCol(I):
    ''' Calculate difference between columns in the matrix

        I : Matrix to operate one

        Finds the absolute element-wise difference between one column and the next.

        diff[:,0]= ||I[:,0]-I[:,1]||

        Returns the difference matrix
    '''
    cols = I.shape[1]
    diff = np.zeros(I.shape,dtype=I.dtype)
    for c in range(1,cols):
       diff[:,c-1]=np.abs(I[:,c]-I[:,c-1])
    return diff

def diffRow(I):
    ''' Calculate difference between rows in the matrix

        I : Matrix to operate one

        Finds the absolute element-wise difference between one row and the next.

        diff[0,:]= ||I[0,:]-I[1,:]||

        Returns the difference matrix
    '''
    rows = I.shape[1]
    diff = np.zeros(I.shape,dtype=I.dtype)
    for r in range(1,rows):
       diff[r-1,:]=np.abs(I[r,:]-I[r-1,:])
    return diff

def remCirclesOutBounds(circles,shape):
   ''' Remove the circles form the set of circles whose area goes out of bounds in the image

       circles : Array containing [centre_x,centre_y,radius] for each circle found
       shape : Shape of the data they were found in

       This function checks if given the postion and radius of the circle whether any portion of it
       goes outside the bounds of shape. It performs a logical check of

       Returns filtered list of circles
   '''
   # search for any circles that are out of bounds
   outofbounds_idx = np.where(np.logical_or(
      np.logical_or(circles[0,:,0]-circles[0,:,2]>128,circles[0,:,0]+circles[0,:,2]>128), # checking if any x coordinates are out of bounds
      np.logical_or(circles[0,:,1]-circles[0,:,2]>128,circles[0,:,1]+circles[0,:,2]>128)))[0] # checking if any y coordinates are out of bounds
   # if any circles were found, delete them and return new array
   # Numpy's delete command is performance costly hence the precheck
   if outofbounds_idx.shape[0]>0:
      # if circles is given straight from Hough circles, then its shape is 1xCx3
      # if the user unwraps it, then it is Cx3
      # this if statement simply makes sure that the correct axis is set
      if len(circles.shape)==3:
         return np.delete(circles,outofbounds_idx,axis=1)
      else:
         return np.delete(circles,outofbounds_idx,axis=0)
   # if no circles are to be removed, return original array
   else:
      return circles

def drawHoughCircles(circles,shape,**kwargs):
    ''' Draw the givenn list of circles from the HoughCircles function onto a blank matrix

        circles : Array of circles data returned by OpenCVs HoughCircles function
        shape : Shape of the mask to draw results on. Mask size is (shape,3)
        **kwargs : Additional arguments to customise drawing
            centre_col : color to draw circle centres with, tuple of uint8 values
            bounday_col : color to draw circle boundaries with, tuple of uint8 values
            iter_cols : Each circle is drawn with a unique color and centre color decided by it's order in the matrix.
                        (255/ci,255/ci,255/ci). Centre and boundary color ARE THE SAME
            num_c : Number of circles to draw. By default it draws all of circles. This allows user to select a certain number

        HoughCircles searches for circular objects in a matrix. The function returns an array
        of circles found within the data given the used function parameters. The array is organised as
        a row of tuples describing centre location (x,y) and radius.

        The centres are drawn as 1 radius circles in one color and the bounday in another color. Draws
        boundaries and then centres so the centres can be more clearly seen for large collections of circles

        Return the matrix that has the circles drawn on.
    '''
    # set colors
    if 'centre_col' in kwargs.keys():
        ccol = kwargs["centre_col"]
    else:
        ccol = (0, 100, 100)
        
    if 'boundary_col' in kwargs.keys():
        bcol = kwargs["bounday_col"]
    else:
        bcol = (255, 0, 255)

    if 'num_c' in kwargs.keys():
        len_c = kwargs['num_c']
    else:
        len_c = circles.shape[1]
        
    # blank matrix
    mask = np.zeros((*shape,3),dtype='uint8')
    # iterate through circles array
    # array is [1xCx3]
    for c in range(len_c):
        # draw circle boundary
        cv2.circle(mask, (circles[0,c,0], circles[0,c,1]), circles[0,c,2], bcol, 1)
    for c in range(len_c):
        cv2.circle(mask, (circles[0,c,0], circles[0,c,1]), 1, ccol, 1)
    # return circle result
    return mask   

def sigmoid(x,lower,upper,growth,v,Q):
    """ Generalized sigmoid function

        lower : lower asymptote
        upper : upper asymptote
        growth: growth factor
        v     : affects which asymptote the max growth occurs near
        Q     : related to y(0)
    """
    return lower + (upper-lower)/((1 + Q*np.exp(-growth*x))**(1.0/v))

def depositedEnergy(beta_v,alpha):
    beta,v = beta_v
    return beta/((np.pi*(2*alpha*(20e-6**3.0))**0.5)*(500.0/(v**0.5)))

# path to footage
path = "D:/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"
pxml = "D:/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/tree2.xml"

print("Reading in vel data in mm/s")
vel = readXMLData(pxml)[3]
vel = np.array(vel,dtype='float')
vel *= 10e3
# read in peak scanning velocity and scale to mm/s
scan_v = 10.9

print("Reading in video data")
frames_set_array = readH264(path)
rows,cols,depth = frames_set_array.shape

target_f = depth # 1049
pixel_pitch=20.0e-6
# sampling frequency of thermal camera, assumed
fc = 65.0
tc = 1/fc
# data matricies
T = np.zeros(frames_set_array.shape,dtype='float64')
I = np.zeros(frames_set_array.shape,dtype='float64')

print("Evaluating up to ", target_f)
for f in range(0,target_f,1):
    #print("f=",f)
    T[:,:,f] = Qr2Temp(frames_set_array[:,:,f],e_ss,T0)
    if f==0:
        I[:,:,f] = Temp2IApprox(T[:,:,f],T0,Kp_spline,Ds_spline,tc)
    else:
        # prev temp has to be T[:,:,f] to get expected behaviour for Tpeak
        # not sure why
        # if T0 is T[:,:,f-1], then Tpeak rises from 500K to 3200K
        I[:,:,f] = Temp2IApprox(T[:,:,f],T[:,:,f-1],Kp_spline,Ds_ispline,tc)
        
    if np.inf in I[:,:,f]:
        np.nan_to_num(I[:,:,f],copy=False)
print(tt)
# laser radius
r0 = 0.00035 #m
# laser radius in pixels
r0_p = int(np.ceil(r0/pixel_pitch))

print("Reading in data...") 
# read in data as structured data types
local_data = np.genfromtxt("thermal-hough-circle-metrics-local.csv",
                           delimiter=',',
                           dtype='float',
                           names=True)

global_data = np.genfromtxt("thermal-hough-circle-metrics-global.csv",
                            delimiter=',',
                            dtype='float',
                            names=True)
print("Generating difference data...")
# construct difference data dictionaries
local_diff_data = {"Diff. "+k : np.diff(local_data[k]) for k in local_data.dtype.names}
global_diff_data = {"Diff. "+k : np.diff(global_data[k]) for k in global_data.dtype.names}
# diff length
diff_length = local_diff_data['Diff. Best_Circle_Accumulator_Score'].shape[0]
## functions for fitting to the data
# cubic
def cubic(x,a,b,c,d):
    return a*x**3.0 + b*x**2.0 + c*x + d
# quadratic
def quad(x,a,b,c):
    return a*x**2.0 + b*x + c
# rational quad function
def rational_poly(x,a,b,c,d,e,f):
    return (a*x**2.0 + b*x + c)/(d*x**2.0 + e*x + f)
# power exponent
def exponent(x,a,b):
    return a*np.exp(x)+b
# arcsin, inverse sine
def arcsin(x,mag,xshift,yshift):
    return x*np.arcsin(x+xshift)+yshift

##print("Fitting functions")
##print("Fitting to K vs Delta Accum")
### fitting a cubic to relationship between change in accumulator score to K area
##popt,pcov = curve_fit(cubic,
##                      local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length], # thermal conductivity of the area
##                      local_diff_data['Diff. Best_Circle_Accumulator_Score']) # change in accumulator score
##
#### filtering data
##data = local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length]
### find standard deviation
##data_std = np.std(data)
### find mean of the data
##data_m = np.mean(data)
### find the data within three standard deviations of the mean
##stdi = np.where( (data>=(data_m-3*data_std)) & (data<=(data_m+3*data_std)))
##data_filt = data[stdi]
### fit cubic to the data
##popt_filt,pcov_filt = curve_fit(cubic,data_filt,
##                                local_diff_data['Diff. Best_Circle_Accumulator_Score'][stdi])
##
### filter based on location of power density peaks
##I_max = I.max(axis=(0,1))
##peaks,_ = find_peaks(I_max)
##tol = 0.05
##I_abs_max = I_max.max()
### searching for peak values within the tolerancce of the power density max
##pki = peaks[np.where(I_max[peaks]>=(1-tol)*I_abs_max)]
##plt.plot(I_max,'b-',peaks,I_max[peaks],'rx',pki,I_max[pki],'yx')
##plt.gca().set(xlabel='Frame Index',ylabel='Peak Power Density (W/m2)')
##plt.gca().legend(['Original','All','Near Max'])
##plt.suptitle('Peaks Identified in Peak Power Density')
##plt.savefig('marked-peaks-peak-I.png')
### just get the values associated with these peaks
##popt_pk,pcov_pk = curve_fit(cubic,data[pki],
##                            local_diff_data['Diff. Best_Circle_Accumulator_Score'][pki])
##
##popt_pk2,pcov_pk2 = curve_fit(cubic,data[peaks],
##                            local_diff_data['Diff. Best_Circle_Accumulator_Score'][peaks])
##
##print("Plotting results")
#### filtering data by peaks
##fpeak,axpeak = plt.subplots(1,3,sharex=True,sharey=True)
##axpeak[0].plot(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],
##         local_diff_data['Diff. Best_Circle_Accumulator_Score'],'rx')
##axpeak[0].set(ylabel='$\Delta$ Accumulator Score')
##axpeak[0].set_title('Original')
##axpeak[1].plot(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length][peaks],
##         local_diff_data['Diff. Best_Circle_Accumulator_Score'][peaks],'bx')
##axpeak[1].set(xlabel='K Area (W.m.K)')
##axpeak[1].set_title('All Peaks')
##axpeak[2].plot(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length][pki],
##         local_diff_data['Diff. Best_Circle_Accumulator_Score'][pki],'yx')
##axpeak[2].set_title('Near Max,tol={:.1f}%'.format(tol*100))
##fpeak.suptitle('Filtering K vs $\Delta$ Accum. Data\n According to Power Density Peaks')
##fpeak.savefig('filt-data-I-peaks-k-area-change-best-circle-accumulator.png')
##
##f,ax = plt.subplots()
##ax.plot(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],
##         local_diff_data['Diff. Best_Circle_Accumulator_Score'],'rx',
##        local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],
##        cubic(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],*popt_pk),'bx')
##ax.legend(['Data','Fitted Cubic'])
##ax.set(xlabel='Thermal Conductivity of the Area (W.m.K)',ylabel='$\Delta$ Accumulator Score')
##f.suptitle('Thermal Conductivity of the Area vs $\Delta$ Accumulator Score \n Fitted to When Power Density Peaks')
##f.savefig('fitted-filt-peaks-near-max-k-area-change-best-circle-accumulator.png')
##
##ax.clear()
##ax.plot(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],
##         local_diff_data['Diff. Best_Circle_Accumulator_Score'],'rx',
##        local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],
##        cubic(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],*popt_pk2),'bx')
##ax.legend(['Data','Fitted Cubic'])
##ax.set(xlabel='Thermal Conductivity of the Area (W.m.K)',ylabel='$\Delta$ Accumulator Score')
##f.suptitle('Thermal Conductivity of the Area vs $\Delta$ Accumulator Score\n Fitted to When P.D within {:.2f}% of Peak'.format(tol*100))
##f.savefig('fitted-filt-peaks-k-area-change-best-circle-accumulator.png')
##
#### filtering data by std
##ax.clear()
##ax.plot(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],
##         local_diff_data['Diff. Best_Circle_Accumulator_Score'],'rx',
##         data_filt,
##         local_diff_data['Diff. Best_Circle_Accumulator_Score'][stdi],'bx')
##ax.legend(['Original','Filtered'])
##ax.set(xlabel='Thermal Conductivity of the Area (W.m.K)',ylabel='$\Delta$ Accumulator Score')
##f.suptitle('Thermal Conductivity of the Area vs $\Delta$ Accumulator Score\n Filtered for Within 3 Standard Deviations')
##f.savefig('filt-data-k-area-change-best-circle-accumulator.png')
##
##ax.clear()
##ax.plot(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],
##        local_diff_data['Diff. Best_Circle_Accumulator_Score'],'rx',
##        data_filt,
##        cubic(data_filt,*popt_filt),'b*')
##ax.legend(['Data','Fitted Cubic'])
##ax.set(xlabel='Thermal Conductivity of the Area (W.m.K)',ylabel='$\Delta$ Accumulator Score')
##f.suptitle('Thermal Conductivity of the Area vs $\Delta$ Accumulator Score\n Fitted to Data Within 3 Standard Deviations')
##f.savefig('fitted-filt-k-area-change-best-circle-accumulator.png')
## 
#### unfiltered
##ax.clear()
##ax.plot(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],
##        local_diff_data['Diff. Best_Circle_Accumulator_Score'],'rx',
##        local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],
##        cubic(local_data['Thermal_Conductivity_of_Area_WmK'][:diff_length],*popt),'b*')
##ax.legend(['Data','Fitted Cubic'])
##ax.set(xlabel='Thermal Conductivity of the Area (W.m.K)',ylabel='$\Delta$ Accumulator Score')
##f.suptitle('Thermal Conductivity of the Area vs $\Delta$ Accumulator Score, Fitted and Raw')
##f.savefig('fitted-k-area-change-best-circle-accumulator.png')
##   
##print("Fitting to Qr vs R")
##x = local_data['Radiated_Heat_Times_Area_W']
##y = local_data['Best_Circle_Radius_Pixels']
##
### fitting raw
##popt,pcov = curve_fit(quad,x,y)
##
#### filtering radius values
### the radius values are for the circle that gives a power estimate closest to 500W
### when there's little activity, the radius is very large as it needs to be to satisfy the search condition
##filt_r = rows/2
##rf = np.where(y<filt_r)
##
##popt_filt,pcov_filt = curve_fit(quad,x[rf],y[rf])
##
### filtering by std
##data_std = x.std()
##data_m = x.mean()
##stdi = np.where( (x>=(data_m-3*data_std)) & (x<=(data_m+3*data_std)))
##popt_filt2,pcov_filt2 = curve_fit(quad,x[stdi],y[stdi])
##
### using peak information
##popt_peak,pcov_peak = curve_fit(quad,x[pki],y[pki])
##popt_peak2,pcov_peak2 = curve_fit(quad,x[peaks],y[peaks])
##
#### choosing limits as reference points for quad
### using the filtered dataset to avoid outliers
##xstd = x[stdi]
##ystd = y[stdi]
### smallest radius
##min_r = ystd.min()
### location of smallest radius
##min_ri = ystd.argmin()
### least heat rad
###min_qr = xstd.min()
##min_qr = np.sort(xstd)[1]
### location of lest heat rad
###min_qri = xstd.argmin()
##min_qri = np.argsort(xstd)[1]
### the max value is the outlier
### second value is the limit 
##max_qr = xstd.max()
##max_qri = xstd.argmax()
#### compile to reference points
##x_ref = [min_qr,xstd[min_ri],max_qr]
##y_ref = [ystd[min_qri],min_r,ystd[max_qri]]
### fitting curve using these data points
##popt_ref,pcov_ref = curve_fit(quad,x_ref,y_ref)
##
### filtering using limit and std info
##data_std = x[rf].std()
##data_m = x[rf].mean()
##stdi_lim = np.where((x[rf]>=(data_m-3*data_std)) & (x[rf]<=(data_m+3*data_std)))
##popt_filt3,pcov_filt3 = curve_fit(quad,x[rf][stdi_lim],y[rf][stdi_lim])
##
### fit using rational eq based on 
##popt_rat,pcov_rat = curve_fit(rational_poly,x[rf][stdi_lim],y[rf][stdi_lim])
##
##print("Fitting Qr to dist")
##dist = local_data['Best_Circle_Distance_from_Centre_Pixels']
### fit to raw values
##popt_exp,pcov_exp = curve_fit(exponent,x,dist)
### using std
##popt_exp2,pcov_exp2 = curve_fit(exponent,x[stdi],dist[stdi])
##
##
##print("Plotting results")
##ax.clear()
##ax.plot(x,y,'rx',x[rf],y[rf],'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels))')
##ax.legend(['Data','r<{0}'.format(filt_r)])
##f.suptitle('Radiated Heat vs Best Circle Radius Filtered Based \n On Overly Large Circles')
##f.savefig('filt-data-half-img-width-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x[stdi],y[stdi],'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels))')
##ax.legend(['Data','3 std.'])
##f.suptitle('Radiated Heat vs Best Circle Radius Filtered Within \n 3 Standard Deviations')
##f.savefig('filt-data-std-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,dist,'rx',x[stdi],dist[stdi],'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Centre Distance (Pixels))')
##ax.legend(['Data','3 std.'])
##f.suptitle('Radiated Heat vs Best Circle Distance From Centre \nFiltered Within 3 Standard Deviations')
##f.savefig('filt-data-std-heat-rad-best-circle-centre-dist.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x[rf][stdi_lim],y[rf][stdi_lim],'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels))')
##ax.legend(['Data','3 std. & r<{0}'.format(filt_r)])
##f.suptitle('Radiated Heat vs Best Circle Radius Filtered\n to Within 3 Standard Deviations and r<{0}'.format(filt_r))
##f.savefig('filt-data-half-img-width-std-heat-rad-best-circle-radius.png')
##
##for a in axpeak:
##   a.clear()
##axpeak[0].plot(x,y,'rx')
##axpeak[0].set(ylabel='Best Circle Radius (Pixels)')
##axpeak[0].set_title('Original')
##axpeak[1].plot(x[peaks],y[peaks],'bx')
##axpeak[1].set(xlabel='Radiated Heat of Area (W)')
##axpeak[1].set_title('Peaks')
##axpeak[2].plot(x[pki],y[pki],'yx')
##axpeak[2].set_title('Near Max,tol={:.1f}%'.format(tol*100))
##fpeak.suptitle('Radiated Heat vs Best Circle Radius Filtered Based \n On Power Density Peaks')
##fpeak.savefig('filt-data-peaks-width-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x,quad(x,*popt),'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels)')
##ax.legend(['Data','$x^2$'])
##f.suptitle('Radiated Heat vs Best Circle Radius, Raw and Fitted')
##f.savefig('fitted-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x,rational_poly(x,*popt_rat),'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels)')
##ax.legend(['Data','Rational (Quad)'])
##f.suptitle('Radiated Heat vs Best Circle Radius \n Rational Eq. (Quadratic) Fitted')
##f.savefig('fitted-rational-filt-half-img-width-std-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x,quad(x,*popt_ref),'bx',x_ref,y_ref,'yo')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels)')
##ax.legend(['Data','$x^2$'])
##f.suptitle('Radiated Heat vs Best Circle Radius \n Quadratic Fitted to Reference Points')
##f.savefig('fitted-ref-filt-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x,quad(x,*popt_filt),'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels)')
##ax.legend(['Data','$x^2$'])
##f.suptitle('Radiated Heat vs Best Circle Radius\nFitted to Radius Values r<{}'.format(filt_r))
##f.savefig('fitted-r0-filt-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x,quad(x,*popt_filt2),'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels)')
##ax.legend(['Data','$x^2$'])
##f.suptitle('Radiated Heat vs Best Circle Radius\n Fitted to Data within 3 Standard Deviations')
##f.savefig('fitted-filt-std-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x,quad(x,*popt_filt3),'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels)')
##ax.legend(['Data','$x^2$'])
##f.suptitle('Radiated Heat vs Best Circle Radius Fitted to Data\n Where r<{0} p and within 3 Standard Deviations'.format(filt_r))
##f.savefig('fitted-filt-half-img-width-std-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x,quad(x,*popt_peak),'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels)')
##ax.legend(['Data','$x^2$'])
##f.suptitle('Radiated Heat vs Best Circle Radius\n Fitted to Where Power Density Peaks')
##f.savefig('fitted-filt-peak-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,y,'rx',x,quad(x,*popt_peak2),'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Radius (Pixels)')
##ax.legend(['Data','$x^2$'])
##f.suptitle('Radiated Heat vs Best Circle Radius\n Fitted to Where P.D. Within {:.2f}% of Max'.format(tol*100))
##f.savefig('fitted-filt-near-max-heat-rad-best-circle-radius.png')
##
##ax.clear()
##ax.plot(x,dist,'rx',x,exponent(x,*popt_exp),'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Centre Distance (Pixels)')
##ax.legend(['Data','ax^b'])
##f.suptitle('Radiated Heat vs Best Circle Centre Distance\n Fitted with an Exponent')
##f.savefig('fitted-heat-rad-best-circle-centre-dist.png')
##
##ax.clear()
##ax.plot(x,dist,'rx',x,exponent(x,*popt_exp2),'bx')
##ax.set(xlabel='Radiated Heat (W)',ylabel='Best Circle Centre Distance (Pixels)')
##ax.legend(['Data','ax^b'])
##f.suptitle('Radiated Heat vs Best Circle Centre Distance\n Fitted to Data within 3 Standard Deviations')
##f.savefig('fitted-filt-std-heat-rad-best-circle-centre-distance.png')
##
#### converting x and y positions to polar coordinates
### read in position and convert to 
##x = local_data["Best_Circle_X_Position_Pixels"]-64.0
##y = local_data["Best_Circle_Y_Position_Pixels"]-64.0
##dist = local_data['Best_Circle_Distance_from_Centre_Pixels']
### convert x,y to arctan
##angle = np.arctan(np.abs(y)/np.abs(x))
### account for different quadrant angles
##angle[(y<0)&(x<0)] += np.pi
##angle[(y<0)&(x>0)] *= -1
##dist[(y<0)|(x<0)] *= -1
##
### for each name in header, collect the ones not containing a sub string
##avoid = ['X','Y','Radius','Dist']
### filename shorthand for each dataset
##short_str = ['rad-heat','k-area','d-area','temp-area','best-circle-radius',
##             'best-circle-acc','best-circle-centre-x','best-circle-centre-y',
##             'best-circle-centre-distance']
### get the header names not containing the substrings in the avoid list
##plot_names = [(ni,name) for (ni,name) in enumerate(local_data.dtype.names) if all([not av in name for av in avoid])]
##
##f,ax = plt.subplots()
##print("Plotting polar coordinates")
##ax.clear()
##ax.plot(angle)
##ax.set(xlabel='Frame Index',ylabel='Angular Position (Radians)')
##f.suptitle('Angular Position of the Best Circle Centre In Radians \n Relative to Image Centre with Positive X To The Right')
##f.savefig('data-polar-coords-angle.png')
##
##ax.clear()
##ax.plot(angle*(180/np.pi))
##ax.set(xlabel='Frame Index',ylabel='Angular Position (Degrees)')
##f.suptitle('Angular Position of the Best Circle Centre In Degrees \n Relative to Image Centre with Positive X To The Right')
##f.savefig('data-polar-coords-angle-degs.png')
##
##ax.clear()
##ax.plot(dist)
##ax.set(xlabel='Frame Index',ylabel='Distance From Centre (Pixels)')
##f.suptitle('Polar Distance of the Best Circle Centre from Image Centre \n Taking into Acount X and Y Position, (X +ve Right)')
##f.savefig('data-polar-coords-dist.png')
##
### plot angular coordinates against variables not related to circle centre
##for name in plot_names:
##   ax.clear()
##   ax.plot(local_data[name[1]],angle,'rx')
##   ax.set(xlabel=name[1].replace('_',' '),ylabel='Angular Position (Radians)')
##   f.suptitle('{} vs Angular Position of \nthe Best Circle Centre'.format(name[1].replace('_',' ')))
##   f.savefig('data-{}-polar-coords-angle.png'.format(short_str[name[0]]))
##
##   ax.clear()
##   ax.plot(local_data[name[1]],dist,'rx')
##   ax.set(xlabel=name[1].replace('_',' '),ylabel='Angular Position (Radians)')
##   f.suptitle('{} vs Polar Distance of \nthe Best Circle Centre from Image Centre'.format(name[1].replace('_',' ')))
##   f.savefig('data-{}-polar-coords-dist.png'.format(short_str[name[0]]))
##   

## fit limits
qr = local_data['Radiated_Heat_Times_Area_W']
x = local_data["Best_Circle_X_Position_Pixels"]
y = local_data["Best_Circle_Y_Position_Pixels"]

def filt3stdIdx(x):
   return np.where((x>=(x.mean()-3*x.std())) & (x<=(x.mean()+3*x.std())))[0]

stdi = filt3stdIdx(qr)

x_stdi = filt3stdIdx(x)

f,ax = plt.subplots()
### break the data into strips
##os.makedirs('StripsFitting',exist_ok=True)
##for num_strips in range(10,50):
##   print("Generating plots for {} strips\r".format(num_strips),end='')
##   # create strips
##   qr_strips = np.linspace(qr[stdi].min(),qr[stdi].max(),num_strips)
##   # lists for storing min max values
##   xs_min = []
##   xs_max = []
##   # find max and min value within each strip
##   for qi in range(0,qr_strips.shape[0]-1):
##      #print(qi,qi+1)
##      # get index of qr vals within strip
##      i = np.where((qr[stdi]>=qr_strips[qi]) & (qr[stdi]<=qr_strips[qi+1]))
##      # find min max x values within strip
##      xs_min.append(x[stdi][i].min())
##      xs_max.append(x[stdi][i].max())
##   # get the middle values between strips
##   qrs = (qr_strips[1:]+qr_strips[:-1])/2
##
##   ## plot result
##   ax.clear()
##   ax.plot(qr[stdi],x[stdi],'rx',# plot data
##            qrs,xs_min,'mo-', # plot x min
##            qrs,xs_max,'ko-') # plot x max
##
##   # add lines representing the strip limits
##   ylim = ax.get_ybound()
##   for q in qr_strips:
##      ax.add_line(Line2D([q]*5,np.linspace(ylim[0],ylim[1],num=5),
##                         color='b',
##                         linestyle='--'))
##
##   ax.legend(['Data','Strip Min','Strip Max','Strips'])
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle X Position (Pixels)')
##   f.suptitle('Radiated Heat Times Area vs Best Circle X Position, Strips={}'.format(num_strips))
##   f.savefig('StripsFitting/strips-fit-heat-rad-best-circle-centre-x-s{}.png'.format(num_strips))

print("Generating density plots")
os.makedirs("DensityPlots",exist_ok=True)
### density plot
##ax.clear()
##xy = np.vstack((qr,x))
### kernel density estimation
##z = gaussian_kde(xy)(xy)
##ax.scatter(qr,x,c=z,edgecolor='')
##ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle X Position (Pixels)')
##f.suptitle('Kernel Density Estimate of Radiated Heat Times Area vs \n Best Circle X Position')
##f.savefig('DensityPlots/density-heat-rad-best-circle-centre-x.png')
##
##dkeys = ['Best_Circle_Accumulator_Score','Best_Circle_Radius_Pixels','Best_Circle_X_Position_Pixels', 'Best_Circle_Y_Position_Pixels', 'Best_Circle_Distance_from_Centre_Pixels']
##div = make_axes_locatable(ax)
##cax = div.append_axes("right",size="5%",pad="2%")
##for k in dkeys:
##   ax.clear()
##   x = local_data[k]
##   xy = np.vstack((qr,x))
##   # kernel density estimation
##   z = gaussian_kde(xy)(xy)
##   st = ax.scatter(qr,x,c=z,edgecolor='')
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='{}'.format(k.replace('_',' ')))
##   f.suptitle('Kernel Density Estimate of Radiated Heat Times Area vs \n {}'.format(k.replace('_',' ')))
##   colorbar(st,cax=cax)
##   f.savefig('DensityPlots/density-heat-rad-{}.png'.format(k.replace('_','-')))
##
##for k in dkeys:
##   ax.clear()
##   x = local_data[k]
##   xy = np.vstack((qr[stdi],x[stdi]))
##   # kernel density estimation
##   z = gaussian_kde(xy)(xy)
##   st = ax.scatter(qr[stdi],x[stdi],c=z,edgecolor='')
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='{}'.format(k.replace('_',' ')))
##   f.suptitle('Kernel Density Estimate of Radiated Heat Times Area vs \n{} within 3 Standard Deviations'.format(k.replace('_',' ')))
##   colorbar(st,cax=cax)
##   f.savefig('DensityPlots/density-heat-rad-3std-{}.png'.format(k.replace('_','-')))
##
##for k in dkeys:
##   ax.clear()
##   x = local_data[k]
##   xy = np.vstack((qr[stdi],x[stdi]))
##   # log of kernel density estimation
##   z = gaussian_kde(xy).logpdf(xy)
##   st = ax.scatter(qr[stdi],x[stdi],c=z,edgecolor='')
##   ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='{}'.format(k.replace('_',' ')))
##   f.suptitle('Log of Kernel Density Estimate of Radiated Heat Times Area vs \n{} within 3 Standard Deviations'.format(k.replace('_',' ')))
##   colorbar(st,cax=cax)
##   f.savefig('DensityPlots/density-log-heat-rad-3std-{}.png'.format(k.replace('_','-')))

os.makedirs("ProbabilityPlots",exist_ok=True)
div = make_axes_locatable(ax)
cax = div.append_axes("right",size="5%",pad="2%")
prob_array = np.zeros((rows,cols))
# generate the KDEs for the different directions
qx_kde = gaussian_kde(np.vstack((qr,x)))
qy_kde = gaussian_kde(np.vstack((qr,y)))
qxy_kde = gaussian_kde(np.vstack((qr,x,y)))

### create 3d plot of kde
##f3d = plt.figure()
##ax3d = f3d.add_subplot(111,projection='3d')
##ax3d.scatter(x,y,qr,c=qxy_kde([qr,x,y]).flat)
##ax3d.set(xlabel='X Position (Pixels)',ylabel='Y Position (Pixels)',zlabel='Radiated Heat Times Area (W)')
##f3d.suptitle('Kernel Density Estimate of Radiated Heat Times Area vs \n Best Circle Centre Position')
##f3d.savefig('DensityPlots/density-3d-heat-rad-best-circle-centre-x-y.png')
##
##ax3d.clear()
##ax3d.scatter(x,y,qr,c=qxy_kde.logpdf([qr,x,y]).flat)
##ax3d.set(xlabel='X Position (Pixels)',ylabel='Y Position (Pixels)',zlabel='Radiated Heat Times Area (W)')
##f3d.suptitle('Log of Kernel Density Estimate of Radiated Heat Times Area vs \n Best Circle Centre Position')
##f3d.savefig('DensityPlots/density-log-3d-heat-rad-best-circle-centre-x-y.png')
### close resources
##plt.close(f3d)

# build coordinate data
xx,yy = np.meshgrid(np.arange(0,rows,1),np.arange(0,cols,1))
# unravel so it can be fed in
# saves repeated ravel calls
xx = xx.ravel()
yy = yy.ravel()
## 1d version for 2d runs
x2 = np.arange(0,rows,1)
y2 = np.arange(0,cols,1)
# results matricies
pk_x = np.zeros(I.shape[2])
pk_y = np.zeros(I.shape[2])
pk_p = np.zeros(I.shape[2])
# super mat

def plotSave(data,ff):
   f2,ax2 = plt.subplots()
   div = make_axes_locatable(ax2)
   cax = div.append_axes("right",size="5%",pad="2%")
   ct = ax2.contourf(prob_array)
   colorbar(ct,cax=cax)
   f2.savefig('ProbabilityPlots\prob-x-y-f{}.png'.format(ff))
   plt.close(f2)
##
##print("Generating probability plots for 3D PDF")
##with h5py.File("gaussian-kde-best-circle-centre-arrow-shape.hdf5",'w') as file:
##   for ff in range(I.shape[2]):
##      print(ff)
##      # create matrix of single value representing the Qr area for all positions
##      qq = np.full((rows,cols),qr[ff])
##      # unravel qq and feed it in to calculate P(x,y | Qr)
##      # reshape result back into 
##      prob_array = qxy_kde(np.vstack((qq.ravel(),xx,yy))).reshape(qq.shape)
##      # find the location of the highest peak
##      pk_x[ff],pk_y[ff] = np.unravel_index(prob_array.argmax(),prob_array.shape)
##      # get the highest prob peak
##      pk_p[ff] = prob_array.max(axis=(0,1))
##      # plot result
##      plotSave(prob_array,ff)
##      # store probability results in a super matrix within a hdf5 file
##      if ff==0:
##         dset = file.create_dataset("P-xy-Qr",(*prob_array.shape,1),prob_array.dtype,
##                                    prob_array, # initial data
##                                    maxshape=(*prob_array.shape,None), # resizable dataset
##                                    compression='gzip',compression_opts=9) # max compression
##      else:
##         dset = file["P-xy-Qr"]
##         dset.resize(dset.shape[2]+1,axis=2) # extend dataset
##         dset[:,:,-1]=prob_array # update last frame
##
##      ax.clear()
##   ax.plot(pk_x)
##   ax.set(xlabel='Frame Index',ylabel='Estimated X Position of Circle Centre (Pixels)')
##   f.suptitle('X Position With the Highest PDF Score')
##   f.suptitle('prob-kde-filt-best-circle-centre-x.png')
##
##   ax.clear()
##   ax.plot(pk_y)
##   ax.set(xlabel='Frame Index',ylabel='Estimated Y Position of Circle Centre (Pixels)')
##   f.suptitle('Y Position With the Highest PDF Score')
##   f.suptitle('prob-kde-filt-best-circle-centre-y.png')
##
##   ax.clear()
##   ax.plot(pk_p)
##   ax.set(xlabel='Frame Index',ylabel='Highest Log PDF Value')
##   f.suptitle('Highest PDF Score over Time for Qr vs (x,y)')
##   f.suptitle('peak-prob-kde-best-circle-centre.png')

##print("Calculating probabilities for separate kdes")
##with h5py.File("gaussian-kde-best-circle-centre-x-y-arrow-shape.hdf5",'w') as file:
##   for ff in range(I.shape[2]):
##      print(ff)
##      qq = np.full(cols,qr[ff])
##      prob_x = qx_kde(np.vstack((qq,x2)))
##      qq = np.full(rows,qr[ff])
##      prob_y = qy_kde(np.vstack((qq,y2)))
##
##      if ff==0:
##         dset_x = file.create_dataset("P-x-Qr",(*prob_x.shape,1),prob_x.dtype,
##                                    prob_x,
##                                    maxshape = (*prob_x.shape,None),
##                                    compression="gzip",compression_opts=9)
##         dset_y = file.create_dataset("P-y-Qr",(*prob_y.shape,1),prob_y.dtype,
##                                    prob_y,
##                                    maxshape = (*prob_y.shape,None),
##                                    compression="gzip",compression_opts=9)
##      else:
##         dset_x = file["P-x-Qr"]
##         dset_x.resize(dset_x.shape[1]+1,axis=1)
##         dset_x[:,-1]=prob_x
##         dset_y = file["P-y-Qr"]
##         dset_y.resize(dset_y.shape[1]+1,axis=1)
##         dset_y[:,-1]=prob_y
##
##   dset_y = file["P-y-Qr"]
##   print("dset x size: ",dset_x.shape)
##   print("dset y size: ",dset_y.shape)
##   print("Generating plot")
##   ## contour plots
##   fp,axp = plt.subplots()
##   divp = make_axes_locatable(axp)
##   caxp = divp.append_axes("right",size="5%",pad="2%")
##   ct = axp.contourf(file["P-x-Qr"])
##   colorbar(ct,cax=caxp)
##   axp.set(xlabel="Heat Radiated Over Area (W)",ylabel="X Position (Pixels)")
##   fp.suptitle("Probability of Circle Centre being in X Position\n given Heat Radiated Over Area")
##   fp.savefig('prob-heat-rad-best-circle-centre-x-arrow-shape.png')
##
##   axp.clear()
##   ct = axp.contourf(file["P-y-Qr"])
##   colorbar(ct,cax=caxp)
##   axp.set(xlabel="Heat Radiated Over Area (W)",ylabel="Y Position (Pixels)")
##   fp.suptitle("Probability of Circle Centre being in Y Position\n given Heat Radiated Over Area")
##   fp.savefig('prob-heat-rad-best-circle-centre-y-arrow-shape.png')
##   plt.close(fp)

print("Generating histograms")
## plot histograms of pdf and log pdf
# X position
z = qx_kde(np.vstack((qr,x)))
z_log = qx_kde.logpdf(np.vstack((qr,x)))
pop,edges = np.histogram(z_log,bins=5)
fh,axh = plt.subplots()
axh.bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
axh.set(xlabel='Log PDF of Radiative Heat Area vs Best Pos. X',ylabel = 'Population')
fh.suptitle('Histogram of Log PDF of Radiative Heat \nArea vs Best Position X')
fh.savefig('hist-log-pdf-qr-vs-best-circle-centre-x.png')

pop,edges = np.histogram(z,bins=5)
axh.clear()
axh.bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
axh.set(xlabel='Log PDF of Radiative Heat Area vs Best Pos. X',ylabel = 'Population')
fh.suptitle('Histogram of PDF of Radiative Heat \nArea vs Best Position X')
fh.savefig('hist-pdf-qr-vs-best-circle-centre-x.png')

# Y position
z = qy_kde(np.vstack((qr,y)))
z_log = qy_kde.logpdf(np.vstack((qr,y)))
pop,edges = np.histogram(z_log,bins=5)
axh.clear()
axh.bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
axh.set(xlabel='Log PDF of Radiative Heat Area vs Best Pos. Y',ylabel = 'Population')
fh.suptitle('Histogram of Log PDF of Radiative Heat \nArea vs Best Position Y')
fh.savefig('hist-log-pdf-qr-vs-best-circle-centre-y.png')

pop,edges = np.histogram(z,bins=5)
axh.clear()
axh.bar(edges[:-1]+(edges[1]-edges[0])/2,pop,width=(edges[1]-edges[0]))
axh.set(xlabel='Log PDF of Radiative Heat Area vs Best Pos. Y',ylabel = 'Population')
fh.suptitle('Histogram of PDF of Radiative Heat \nArea vs Best Position Y')
fh.savefig('hist-pdf-qr-vs-best-circle-centre-y.png')
plt.close(fh)

# filter data to just power peak values
I_max = I.max(axis=(0,1))
print("I_max shape ",I_max.shape)
peaks,_ = find_peaks(I_max)
tol = 0.05
I_abs_max = I_max.max()
# searching for peak values within the tolerancce of the power density max
pki = peaks[np.where(I_max[peaks]>=(1-tol)*I_abs_max)]

##print("Generating plots for filtered density peak data")
### generate plots showing which data points are still in use
##fd,axd = plt.subplots()
##axd.plot(qr,x,'rx',qr[pki],x[pki],'bx')
##axd.set(xlabel='Radiative Heat Times Area (W)',ylabel='Best Circle Centre X Position (Pixels)')
##axd.legend(['Original','Peaks'])
##fd.suptitle('Radiative Heat Times Area vs Best Centre X Position\n Filtered by Power Density Peaks')
##fd.savefig('filt-data-I-peaks-heat-rad-area-best-circle-centre-x.png')
##
##axd.clear()
##axd.plot(qr,y,'rx',qr[pki],y[pki],'bx')
##axd.set(xlabel='Radiative Heat Times Area (W)',ylabel='Best Circle Centre Y Position (Pixels)')
##axd.legend(['Original','Peaks'])
##fd.suptitle('Radiative Heat Times Area vs Best Centre Y Position\n Filtered by Power Density Peaks')
##fd.savefig('filt-data-I-peaks-heat-rad-area-best-circle-centre-y.png')

# fit kdes to filtered data
qx_pk_kde = gaussian_kde(np.vstack((qr[pki],x[pki])))
qy_pk_kde = gaussian_kde(np.vstack((qr[pki],y[pki])))
qxy_pk_kde = gaussian_kde(np.vstack((qr[pki],x[pki],y[pki])))

##print("Generating density plots for filtered data")
#### generate density plots for the filtered data
##ax.clear()
##st = ax.scatter(qr[pki],x[pki],c=qx_pk_kde(np.vstack((qr[pki],x[pki]))),edgecolor='')
##ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Centre X Position (Pixels)')
##f.suptitle('Kernel Density Estimate of Radiated Heat Times Area vs \nBest Circle Centre Position X within 5% of P.D Peak')
##colorbar(st,cax=cax)
##f.savefig('DensityPlots/density-filt-I-pk-heat-rad-best-circle-centre-x.png')
##
##ax.clear()
##st = ax.scatter(qr[pki],y[pki],c=qy_pk_kde(np.vstack((qr[pki],y[pki]))),edgecolor='')
##ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Centre Y Position (Pixels)')
##f.suptitle('Kernel Density Estimate of Radiated Heat Times Area vs \nBest Circle Centre Position Y within 5% of P.D Peak')
##colorbar(st,cax=cax)
##f.savefig('DensityPlots/density-filt-I-pk-heat-rad-best-circle-centre-y.png')
##
##ax.clear()
##st = ax.scatter(qr[pki],x[pki],c=qx_pk_kde.logpdf(np.vstack((qr[pki],x[pki]))),edgecolor='')
##ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Centre X Position (Pixels)')
##f.suptitle('Log of Kernel Density Estimate of Radiated Heat Times Area vs \nBest Circle Centre Position X within 5% of P.D Peak')
##colorbar(st,cax=cax)
##f.savefig('DensityPlots/density-filt-I-pk-log-heat-rad-best-circle-centre-x.png')
##
##ax.clear()
##st = ax.scatter(qr[pki],y[pki],c=qy_pk_kde.logpdf(np.vstack((qr[pki],y[pki]))),edgecolor='')
##ax.set(xlabel='Radiated Heat Times Area (W)',ylabel='Best Circle Centre Y Position (Pixels)')
##f.suptitle('Log of Kernel Density Estimate of Radiated Heat Times Area vs \nBest Circle Centre Position Y within 5% of P.D Peak')
##colorbar(st,cax=cax)
##f.savefig('DensityPlots/density-filt-I-pk-log-heat-rad-best-circle-centre-y.png')

def plotSave(data,ff):
   f2,ax2 = plt.subplots()
   div = make_axes_locatable(ax2)
   cax = div.append_axes("right",size="5%",pad="2%")
   ct = ax2.contourf(prob_array)
   colorbar(ct,cax=cax)
   f2.savefig('ProbabilityPlots\prob-peak-filt-x-y-f{}.png'.format(ff))
   plt.close(f2)

print("Calculating probabilities for separate kdes using peak data")
with h5py.File("gaussian-kde-peak-best-circle-centre-x-y-arrow-shape.hdf5",'w') as file:
   for ff in range(I.shape[2]):
      print(ff)
      qq = np.full(cols,qr[ff])
      prob_x = qx_kde(np.vstack((qq,x2)))
      qq = np.full(rows,qr[ff])
      prob_y = qy_kde(np.vstack((qq,y2)))

      if ff==0:
         dset_x = file.create_dataset("P-x-Qr",(*prob_x.shape,1),prob_x.dtype,
                                    prob_x,
                                    maxshape = (*prob_x.shape,None),
                                    compression="gzip",compression_opts=9)
         dset_y = file.create_dataset("P-y-Qr",(*prob_y.shape,1),prob_y.dtype,
                                    prob_y,
                                    maxshape = (*prob_y.shape,None),
                                    compression="gzip",compression_opts=9)
      else:
         dset_x = file["P-x-Qr"]
         dset_x.resize(dset_x.shape[-1]+1,axis=1)
         dset_x[:,-1]=prob_x
         dset_x = file["P-y-Qr"]
         dset_y.resize(dset_y.shape[-1]+1,axis=1)
         dset_y[:,-1]=prob_y
   print("Generating plot")
   ax.clear()
   div = make_axes_locatable(ax)
   cax = div.append_axes("right",size="5%",pad="2%")
   ct = ax.contourf(file["P-x-Qr"])
   colorbar(ct,cax=cax)
   ax.set(xlabel="Heat Radiated Over Area (W)",ylabel="X Position (Pixels)")
   f.suptitle("Probability of Circle Centre being in X Position\n given Heat Radiated Over Area")
   f.savefig('prob-filt-peak-heat-rad-best-circle-centre-x-arrow-shape.png')

   ax.clear()
   ct = ax.contourf(file["P-y-Qr"])
   colorbar(ct,cax=cax)
   ax.set(xlabel="Heat Radiated Over Area (W)",ylabel="Y Position (Pixels)")
   f.suptitle("Probability of Circle Centre being in Y Position\n given Heat Radiated Over Area")
   f.savefig('prob-filt-peak-heat-rad-best-circle-centre-y-arrow-shape.png')


print("Running KDEs for peak filtered data")
print("Generating probability plots for 3D PDF")
with h5py.File("gaussian-kde-peak-best-circle-centre-arrow-shape.hdf5",'w') as file:
   for ff in range(I.shape[2]):
      print(ff)
      # create matrix of single value representing the Qr area for all positions
      qq = np.full((rows,cols),qr[ff])
      # unravel qq and feed it in to calculate P(x,y | Qr)
      # reshape result back into 
      prob_array = qxy_pk_kde(np.vstack((qq.ravel(),xx,yy))).reshape(qq.shape)
      # find the location of the highest peak
      pk_x[ff],pk_y[ff] = np.unravel_index(prob_array.argmax(),prob_array.shape)
      # get the highest prob peak
      pk_p[ff] = prob_array.max(axis=(0,1))
      # plot result
      plotSave(prob_array,ff)
      # store probability results in a super matrix within a hdf5 file
      if ff==0:
         dset = file.create_dataset("P-xy-Qr",(*prob_array.shape,1),prob_array.dtype,
                                    prob_array, # initial data
                                    maxshape=(*prob_array.shape,None), # resizable dataset
                                    compression='gzip',compression_opts=9) # max compression
      else:
         dset = file["P-xy-Qr"]
         dset.resize(dset.shape[2]+1,axis=2) # extend dataset
         dset[:,:,-1]=prob_array # update last frame

   ax.clear()
   ax.plot(pk_x)
   ax.set(xlabel='Frame Index',ylabel='Estimated X Position of Circle Centre (Pixels)')
   f.suptitle('X Position With the Highest PDF Score')
   f.suptitle('prob-kde-filt-I-pk-best-circle-centre-x.png')

   ax.clear()
   ax.plot(pk_y)
   ax.set(xlabel='Frame Index',ylabel='Estimated Y Position of Circle Centre (Pixels)')
   f.suptitle('Y Position With the Highest PDF Score')
   f.suptitle('prob-kde-filt-I-pk-best-circle-centre-y.png')

   ax.clear()
   ax.plot(pk_p)
   ax.set(xlabel='Frame Index',ylabel='Highest Log PDF Value')
   f.suptitle('Highest PDF Score over Time')
   f.suptitle('peak-log-filt-I-pk-kde.png')


r = local_data['Best_Circle_Radius_Pixels']
qr_kde = gaussian_kde(np.vstack((qr,r)))
radius_range = np.arange(r.min(),r.max(),1.0)

r_prob = np.zeros(radius_range.shape)
print("Running KDE for laser radius")
with h5py.File("gaussian-kde-best-circle-radius-arrow-shape.hdf5",'w') as file:
   for ff in range(I.shape[2]):
      # should produce a row of probabilities indicating the probability of radius r being correct
      r_prob = qr_kde([[qr[ff]]*radius_range.shape[0],radius_range])
      if ff==0:
            dset = file.create_dataset("P-r-Qr",(*r_prob.shape,1),r_prob.dtype,
                                       r_prob, # initial data
                                       maxshape=(*r_prob.shape,None), # resizable dataset
                                       compression='gzip',compression_opts=9) # max compression
      else:
         dset = file["P-r-Qr"]
         dset.resize(dset.shape[1]+1,axis=1) # extend dataset
         dset[:,-1]=r_prob # update last frame

   ax.clear()
   ct = ax.contourf(file["P-r-Qr"])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   div = make_axes_locatable(ax)
   cax = div.append_axes("right",size="5%",pad="2%")
   colorbar(ct,cax=cax)
   f.suptitle('Probabilty Spread for Laser Radius for Different Frames')
   f.savefig('ProbabilityPlots/prob-kde-laser-radius.png')
   
qr_pk_kde = gaussian_kde(np.vstack((qr[pki],r[pki])))
radius_range = np.arange(r.min(),r.max(),1.0)
with h5py.File("gaussian-kde-peak-best-circle-radius-arrow-shape.hdf5",'w') as file:
   for ff in range(I.shape[2]):
      # should produce a row of probabilities indicating the probability of radius r being correct
      r_prob = qr_kde([[qr[ff]]*radius_range.shape[0],radius_range])
      if ff==0:
            dset = file.create_dataset("P-r-Qr",(*r_prob.shape,1),r_prob.dtype,
                                       r_prob, # initial data
                                       maxshape=(*r_prob.shape,None), # resizable dataset
                                       compression='gzip',compression_opts=9) # max compression
      else:
         dset = file["P-r-Qr"]
         dset.resize(dset.shape[1]+1,axis=1) # extend dataset
         dset[:,-1]=r_prob # update last frame

   ax.clear()
   ct = ax.contourf(file["P-r-Qr"])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   div = make_axes_locatable(ax)
   cax = div.append_axes("right",size="5%",pad="2%")
   colorbar(ct,cax=cax)
   f.suptitle('Probabilty Spread for Laser Radius for Different Frames\nAfter Being Trained on Peak Values')
   f.savefig('prob-kde-peak-laser-radius.png')
   
