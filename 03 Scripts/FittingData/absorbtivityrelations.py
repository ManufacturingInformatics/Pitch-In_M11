from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import color
import os
import cv2
import csv
from itertools import zip_longest

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
path = "C:/Users/uos/Documents/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"
pxml = "C:/Users/uos/Documents/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/tree2.xml"

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

# laser radius
r0 = 0.00035 #m
# laser radius in pixels
r0_p = int(np.ceil(r0/pixel_pitch))
# assuming gaussian behaviour, 99.7% of values are within 4 std devs of mean
# used as an upper limit in hough circles
rmax_gauss = 4*r0_p

print("Reading in absorbtivity parameters")
## read in absorbtivity coefficients
lower_params = np.fromfile("absorb-poly-lower-params.txt")
upper_params = np.fromfile("absorb-poly-upper-params.txt")
growth_params = np.fromfile("absorb-poly-growth-params.txt")
Q_params = np.fromfile("absorb-poly-Q-params.txt")
v_params = np.fromfile("absorb-poly-v-params.txt")

print("Generating polynomials")
# create polynomial objects
lower_poly = poly.Polynomial(lower_params)
upper_poly = poly.Polynomial(upper_params)
growth_poly = poly.Polynomial(growth_params)
Q_poly = poly.Polynomial(Q_params)
v_poly = poly.Polynomial(v_params)

P = 500.0 # W
# get sigmoid parameters
absorb_params = [lower_poly(scan_v),upper_poly(scan_v),growth_poly(scan_v)
                 ,v_poly(scan_v),Q_poly(scan_v)]
# calculate absorbtivity based off
peak_absorb = sigmoid(P,*absorb_params)

## Calculate absorbtivity for entire velocity set
print("Estimating absorbtivity for dataset")
absorb = np.zeros(vel.shape[0])
for ff in range(vel.shape[0]):
    absorb_params = [lower_poly(vel[ff]),upper_poly(vel[ff]),growth_poly(vel[ff])
                 ,Q_poly(vel[ff]),v_poly(vel[ff])]
    absorb[ff] = sigmoid(P,*absorb_params)
    
f,ax = plt.subplots()
ax.plot(absorb)
ax.set(xlabel='Data Index', ylabel='Predicted Absorbtivity')
f.suptitle('Predicted Absorbtivity Using Velocity Data')
f.savefig('absorb-predict-xml-data.png')

print("Calculating total radiated heat")
Qr_total = np.zeros(I.shape[2],dtype='float64')
Qr_total_area = np.zeros(I.shape[2],dtype='float64')
Qr_mat = np.zeros(I.shape[2],dtype='float64')

mat_T = np.zeros(I.shape[2],dtype='float64')

frame_area = (128*128)*pixel_pitch**2.0
for ff in range(frames_set_array.shape[2]):
    Qr_total[ff] = np.sum(frames_set_array[:,:,ff],dtype='float64')
    Qr_total_area[ff] = Qr_total[ff]*frame_area
    if Qr_total_area[ff]>50.0:
        Qr_mat[ff] = 500.0-Qr_total_area[ff]
    else:
        Qr_mat[ff] = 0.0

    mat_T[ff] = Qr2Temp(Qr_mat[ff],e_ss,T0)
    
ax.clear()
ax.plot(Qr_total)
ax.set(xlabel='Frame Index',ylabel='Total Heat Radiated (W/m2)')
f.suptitle('Total Heat Radiated in Each Frame')
f.savefig('head-rad-total.png')

ax.clear()
ax.plot(Qr_total_area)
ax.set(xlabel='Frame Index',ylabel='Total Heat Radiated for Area(W)')
f.suptitle('Total Heat Radiated in Each Frame Multiplied by Frame Area')
f.savefig('heat-rad-total-area.png')

ax.clear()
ax.plot(Qr_mat)
ax.set(xlabel='Frame Index',ylabel='Total Heat Trapped (W)')
f.suptitle('Total Heat Trapped in the Material')
f.savefig('heat-rad-material.png')

ax.clear()
ax.plot(mat_T)
ax.set(xlabel='Frame Index',ylabel='Estimated Material Temperature (K)')
f.suptitle('Estimated Material Temperature')
f.savefig('material-est-temp.png')

print("trying circles")
# radii range to search for
radii_range = np.arange(30,70,1.0)
# locally normalized matrix
I_norm = I/I.max(axis=(0,1))
# globally normalized matrix
I_global = I/(I.max(axis=(0,1)).max())
# area of the image in physical space
im_area = (rows*cols)*(pixel_pitch**2.0)
# centre of the image
im_centre = [np.floor(rows/2),np.floor(cols/2)]

### stats to collect
## locally equalized data
# radius of the best circle
best_r = np.zeros(I.shape[2])
# accumulator score
best_ac = np.zeros(I.shape[2])
# x position of centre
best_x = np.zeros(I.shape[2])
# y position of centre
best_y = np.zeros(I.shape[2])
# distance of best circle from image centre
best_dist = np.zeros(I.shape[2])

## for globally equalized data
# radius of the best circle
best_rg = np.zeros(I.shape[2])
# accumulator score
best_acg = np.zeros(I.shape[2])
# x position of centre
best_xg = np.zeros(I.shape[2])
# y position of centre
best_yg = np.zeros(I.shape[2])
# distance of best circle from image centre
best_distg = np.zeros(I.shape[2])

# thermal conductivity times area
K_area = np.zeros(I.shape[2])
# thermal diffusivity times area
D_area = np.zeros(I.shape[2])
# total heat in area
Qr_area = np.zeros(I.shape[2],dtype='float64')
# temperature in the area
T_area = np.zeros(I.shape[2],dtype='float64')

for ff in range(I.shape[2]):
    print(ff)
    ## collect information about the raw frames
    # total heat in the area
    Qr_area[ff] = np.sum(frames_set_array[:,:,ff],dtype='float64')*im_area
    # temperature in the area
    T_area[ff] = Qr2Temp(Qr_area[ff],e_ss,T0)
    # thermal conductivity using area value
    K_area[ff] = Kp_spline(T_area[ff])
    # thermal diffusivity using area value
    D_area[ff] = Ds_ispline(T_area[ff])

    ## locally equalized frames
    # find the hough circles in the image
    res = hough_circle(I_norm[:,:,ff],radii_range)
    accums,cx,cy,radii = hough_circle_peaks(res,radii_range)
    # turn into an array that can be passed to list comprehension
    circles = np.array([[[x,y,r] for x,y,r in zip(cx,cy,radii)]],dtype='float32')
    # calculate power for each circle
    pest = np.array([powerEstHoughCircle(c,I[:,:,ff],pixel_pitch) for c in circles[0]])
    # find the best circle
    bestc = np.abs(500.0-pest).argmin()
    ## collect stats
    # radius of the best circle
    best_r[ff] = radii[bestc]
    # accumulator score
    best_ac[ff] = accums[bestc]
    # centre position
    best_x[ff] = cx[bestc]
    best_y[ff] = cy[bestc]

    best_dist[ff] = (np.abs(im_centre[0]-best_x)**2.0 + np.abs(im_centre[1]-best_y)**2.0)**0.5
    
    ## globally equalized frames
    # find the hough circles in the image
    res = hough_circle(I_global[:,:,ff],radii_range)
    accums,cx,cy,radii = hough_circle_peaks(res,radii_range)
    # turn into an array that can be passed to list comprehension
    circles = np.array([[[x,y,r] for x,y,r in zip(cx,cy,radii)]],dtype='float32')
    # calculate power for each circle
    pest = np.array([powerEstHoughCircle(c,I[:,:,ff],pixel_pitch) for c in circles[0]])
    # find the best circle
    bestc = np.abs(500.0-pest).argmin()
    ## collect stats
    # radius of the best circle
    best_rg[ff] = radii[bestc]
    # accumulator score
    best_acg[ff] = accums[bestc]
    # centre position
    best_xg[ff] = cx[bestc]
    best_yg[ff] = cy[bestc]

    best_distg[ff] = ((im_centre[0]-best_xg[ff])**2.0 + (im_centre[1]-best_yg[ff])**2.0)**0.5
    

print("Plotting results...")

ax.clear()
ax.plot(K_area)
ax.set(xlabel='Frame Index',ylabel='Thermal Conductivity (W.mK)')
f.suptitle('Thermal Conductivity Based off the Area Temperature')
f.savefig('area-thermal-conductivity.png')

ax.clear()
ax.plot(T_area)
ax.set(xlabel='Frame Index',ylabel='Area Temperature (K)')
f.suptitle('Equivalent Temperature Based off the Area Radiated Heat')
f.savefig('area-temperature.png')

ax.clear()
ax.plot(D_area*10**6.0)
ax.set(xlabel='Frame Index',ylabel='Thermal Diffusivity (um2/s)')
f.suptitle('Thermal Diffusivity Based off the Area Temperature')
f.savefig('area-thermal-diffusivity.png')

##### LOCAL DATA #####
#//#
ax.clear()
ax.plot(best_x)
ax.set(xlabel='Frame Index',ylabel='Best Centre, X Coordinate (Pixels)')
f.suptitle('X Coordinate of the Centre of the Best Circle (Local)')
f.savefig('best-circle-centre-x.png')

ax.clear()
ax.plot(best_y)
ax.set(xlabel='Frame Index',ylabel='Best Centre, Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of the Centre of the Best Circle (Local)')
f.savefig('best-circle-centre-Y.png')

ax.clear()
ax.plot(np.diff(best_r))
ax.set(xlabel='Frame Index',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Change in Best Circle Radius (Local)')
f.savefig('best-circle-radius-change.png')

ax.clear()
ax.plot(best_r)
ax.set(xlabel='Frame Index',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Best Circle Radius (Local)')
f.savefig('best-circle-radius.png')

ax.clear()
ax.plot(best_dist)
ax.set(xlabel='Frame Index',ylabel='Distance to Centre (Pixels)')
f.suptitle('Distance Between the Best Circle Centre and Centre of Image (Local)')
f.savefig('dist-im-centre-local-best-circle-centre.png')

ax.clear()
ax.plot(best_x,best_y,'rx')
ax.set(xlabel='Best Circle X Position (Pixels)',ylabel='Best Circle Y Position (Pixels)',
       xlim=[0,cols],ylim=[0,rows])
f.suptitle('Coordinates of the Centres of the Best Circles (Local)')
f.savefig('best-circle-centres.png')

## plotting against best circle radius
ax.clear()
ax.plot(Qr_area,best_r,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Heat Radiated and Best Circle Radius (Local)')
f.savefig('heat-rad-best-circle-radius.png')

ax.clear()
ax.plot(T_area,best_r,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Area Temperature and Best Circle Radius (Local)')
f.savefig('temp-area-best-circle-radius.png')

ax.clear()
ax.plot(K_area,best_r,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and Best Circle Radius (Local)')
f.savefig('k-area-best-circle-radius.png')

ax.clear()
ax.plot(D_area,best_r,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and Best Circle Radius (Local)')
f.savefig('d-area-best-circle-radius.png')

## plotting against change in best circle radius
ax.clear()
diff_r = np.diff(best_r)
ax.plot(Qr_area[:diff_r.shape[0]],diff_r,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Heat Radiated and\n Change in Best Circle Radius (Local)')
f.savefig('heat-rad-change-best-circle-radius.png')

ax.clear()
ax.plot(T_area[:diff_r.shape[0]],diff_r,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Area Temperature and\n Change in Best Circle Radius (Local)')
f.savefig('temp-area-change-best-circle-radius.png')

ax.clear()
ax.plot(K_area[:diff_r.shape[0]],diff_r,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\n Change in Best Circle Radius (Local)')
f.savefig('k-area-change-best-circle-radius.png')

ax.clear()
ax.plot(D_area[:diff_r.shape[0]],diff_r,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\n Change in Best Circle Radius (Local)')
f.savefig('d-area-change-best-circle-radius.png')

## plotting change against change in best circle radius
ax.clear()
ax.plot(np.diff(Qr_area),diff_r,'rx')
ax.set(xlabel='Change Heat Radiated (W)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and \n Change in Best Circle Radius (Local)')
f.savefig('change-heat-rad-change-best-circle-radius.png')

ax.clear()
ax.plot(np.diff(T_area),diff_r,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and \n Change in Best Circle Radius (Local)')
f.savefig('change-temp-area-change-best-circle-radius.png')

ax.clear()
ax.plot(np.diff(K_area),diff_r,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity \n and Change in Best Circle Radius (Local)')
f.savefig('change-k-area-change-best-circle-radius.png')

ax.clear()
ax.plot(np.diff(D_area),diff_r,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity \nand Change in Best Circle Radius (Local)')
f.savefig('change-d-area-change-best-circle-radius.png')

## plotting change against best circle radius
ax.clear()
ax.plot(np.diff(Qr_area),best_r[:diff_r.shape[0]],'rx')
ax.set(xlabel='Change Heat Radiated (W)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and \n Change in Best Circle Radius (Local)')
f.savefig('change-heat-rad-change-best-circle-radius.png')

ax.clear()
ax.plot(np.diff(T_area),best_r[:diff_r.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and \n Change in Best Circle Radius (Local)')
f.savefig('change-temp-area-change-best-circle-radius.png')

ax.clear()
ax.plot(np.diff(K_area),best_r[:diff_r.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity \n and Change in Best Circle Radius (Local)')
f.savefig('change-k-area-change-best-circle-radius.png')

ax.clear()
ax.plot(np.diff(D_area),best_r[:diff_r.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity \nand Best Circle Radius (Local)')
f.savefig('change-d-area-best-circle-radius.png')

## plotting against best circle score
ax.clear()
ax.plot(Qr_area,best_ac,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Heat Radiated and\n Best Circle Accumulator Score (Local)')
f.savefig('heat-rad-best-circle-accumulator.png')

ax.clear()
ax.plot(T_area,best_ac,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Area Temperature and\n Best Circle Accumulator Score (Local)')
f.savefig('temp-area-best-circle-accumulator.png')

ax.clear()
ax.plot(K_area,best_ac,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Thermal Conductivity and\n Best Circle Accumulator Score (Local)')
f.savefig('k-area-best-circle-accumulator.png')

ax.clear()
ax.plot(D_area,best_ac,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Thermal Diffusivity and\n Best Circle Accumulator Score (Local)')
f.savefig('d-area-best-circle-accumulator.png')

## plotting against change in best circle score
ax.clear()
diff_ac = np.diff(best_ac)
ax.plot(Qr_area[:diff_ac.shape[0]],diff_ac,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Heat Radiated and\n Change in Best Circle Accumulator Score (Local)')
f.savefig('heat-rad-change-best-circle-accumulator.png')

ax.clear()
ax.plot(T_area[:diff_ac.shape[0]],diff_ac,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Area Temperature and\n Change in Best Circle Accumulator Score (Local)')
f.savefig('temp-area-change-best-circle-accumulator.png')

ax.clear()
ax.plot(K_area[:diff_ac.shape[0]],diff_ac,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Thermal Conductivity and\n Change in Best Circle Accumulator Score (Local)')
f.savefig('k-area-change-best-circle-accumulator.png')

ax.clear()
ax.plot(D_area[:diff_ac.shape[0]],diff_ac,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Thermal Diffusivity and\n Change in Best Circle Accumulator Score (Local)')
f.savefig('d-area-change-best-circle-accumulator.png')

## plotting change against change in best circle score
ax.clear()
ax.plot(np.diff(Qr_area),diff_ac,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Heat Radiated and\n Change in Best Circle Accumulator Score (Local)')
f.savefig('change-heat-rad-change-best-circle-accumulator.png')

ax.clear()
ax.plot(np.diff(T_area),diff_ac,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Area Temperature and\n Change in Best Circle Accumulator Score (Local)')
f.savefig('change-temp-area-change-best-circle-accumulator.png')

ax.clear()
ax.plot(np.diff(K_area),diff_ac,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Thermal Conductivity and\n Change in Best Circle Accumulator Score (Local)')
f.savefig('change-k-area-change-best-circle-accumulator.png')

ax.clear()
ax.plot(np.diff(D_area),diff_ac,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Thermal Diffusivity and\n Change in Best Circle Accumulator Score (Local)')
f.savefig('change-d-area-change-best-circle-accumulator.png')

## plotting change against best circle score
ax.clear()
ax.plot(np.diff(Qr_area),best_ac[:diff_ac.shape[0]],'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Heat Radiated and\nBest Circle Accumulator Score (Local)')
f.savefig('change-heat-rad-best-circle-accumulator.png')

ax.clear()
ax.plot(np.diff(T_area),best_ac[:diff_ac.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Area Temperature and\nBest Circle Accumulator Score (Local)')
f.savefig('change-temp-area-change-best-circle-accumulator.png')

ax.clear()
ax.plot(np.diff(K_area),best_ac[:diff_ac.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Thermal Conductivity and\nBest Circle Accumulator Score (Local)')
f.savefig('change-k-area-best-circle-accumulator.png')

ax.clear()
ax.plot(np.diff(D_area),best_ac[:diff_ac.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nBest Circle Accumulator Score (Local)')
f.savefig('change-d-area-best-circle-accumulator.png')

## plotting against x position of best circle centre
ax.clear()
ax.plot(Qr_area,best_x,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nX Position of Centre of Best Circle (Local)')
f.savefig('heat-rad-best-circle-centre-x.png')

ax.clear()
ax.plot(T_area,best_x,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nX Position of Centre of Best Circle (Local)')
f.savefig('temp-area-best-circle-centre-x.png')

ax.clear()
ax.plot(K_area,best_x,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nX Position of Centre of Best Circle (Local)')
f.savefig('k-area-best-circle-centre-x.png')

ax.clear()
ax.plot(D_area,best_x,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nX Position of Centre of Best Circle (Local)')
f.savefig('d-area-best-circle-centre-x.png')

## plotting against change x position of best circle centre
ax.clear()
diff_x = np.diff(best_x)
ax.plot(Qr_area[:diff_x.shape[0]],diff_x,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('heat-rad-change-best-circle-centre-x.png')

ax.clear()
ax.plot(T_area[:diff_x.shape[0]],diff_x,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('temp-area-change-best-circle-centre-x.png')

ax.clear()
ax.plot(K_area[:diff_x.shape[0]],diff_x,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('k-area-best-change-circle-centre-x.png')

ax.clear()
ax.plot(D_area[:diff_x.shape[0]],diff_x,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('d-area-best-change-circle-centre-x.png')

## plotting change against x position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),best_x[:diff_x.shape[0]],'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nX Position of Centre of Best Circle (Local)')
f.savefig('change-heat-rad-best-circle-centre-x.png')

ax.clear()
ax.plot(np.diff(T_area),best_x[:diff_x.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nX Position of Centre of Best Circle (Local)')
f.savefig('change-temp-area-best-circle-centre-x.png')

ax.clear()
ax.plot(np.diff(K_area),best_x[:diff_x.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nX Position of Centre of Best Circle (Local)')
f.savefig('change-k-area-best-circle-centre-x.png')

ax.clear()
ax.plot(np.diff(D_area),best_x[:diff_x.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nX Position of Centre of Best Circle (Local)')
f.savefig('change-d-area-best-circle-centre-x.png')

## plotting change agsinst chhange in x position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),diff_x,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('change-heat-rad-change-best-circle-centre-x.png')

ax.clear()
ax.plot(np.diff(T_area),diff_x,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('change-temp-area-change-best-circle-centre-x.png')

ax.clear()
ax.plot(np.diff(K_area),diff_x,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('change-k-area-change-best-circle-centre-x.png')

ax.clear()
ax.plot(np.diff(D_area),diff_x,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('change-d-area-change-best-circle-centre-x.png')

## plotting change against change x position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),diff_x,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('change-heat-rad-change-best-circle-centre-x.png')

ax.clear()
ax.plot(np.diff(T_area),diff_x,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('change-temp-area-change-best-circle-centre-x.png')

ax.clear()
ax.plot(np.diff(K_area),diff_x,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('change-k-area-change-best-circle-centre-x.png')

ax.clear()
ax.plot(np.diff(D_area),diff_x,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nChange in X Position of Centre of Best Circle (Local)')
f.savefig('change-d-area-change-best-circle-centre-x.png')

## plotting against y position of best circle centre
ax.clear()
ax.plot(Qr_area,best_y,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nY Position of Centre of Best Circle (Local)')
f.savefig('heat-rad-best-circle-centre-y.png')

ax.clear()
ax.plot(T_area,best_y,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nY Position of Centre of Best Circle (Local)')
f.savefig('temp-area-best-circle-centre-y.png')

ax.clear()
ax.plot(K_area,best_y,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nY Position of Centre of Best Circle (Local)')
f.savefig('k-area-best-circle-centre-y.png')

ax.clear()
ax.plot(D_area,best_y,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nY Position of Centre of Best Circle (Local)')
f.savefig('d-area-best-circle-centre-y.png')

## plotting against change y position of best circle centre
ax.clear()
diff_y = np.diff(best_y)
ax.plot(Qr_area[:diff_y.shape[0]],diff_y,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nChange in Y Position of Centre of Best Circle (Local)')
f.savefig('heat-rad-change-best-circle-centre-y.png')

ax.clear()
ax.plot(T_area[:diff_y.shape[0]],diff_y,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nChange in Y Position of Centre of Best Circle (Local)')
f.savefig('temp-area-change-best-circle-centre-y.png')

ax.clear()
ax.plot(K_area[:diff_y.shape[0]],diff_y,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nChange in Y Position of Centre of Best Circle (Local)')
f.savefig('k-area-best-change-circle-centre-y.png')

ax.clear()
ax.plot(D_area[:diff_y.shape[0]],diff_y,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nChange in Y Position of Centre of Best Circle (Local)')
f.savefig('d-area-best-change-circle-centre-y.png')

## plotting change against y position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),best_y[:diff_y.shape[0]],'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nY Position of Centre of Best Circle (Local)')
f.savefig('change-heat-rad-best-circle-centre-y.png')

ax.clear()
ax.plot(np.diff(T_area),best_y[:diff_y.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nX Position of Centre of Best Circle (Local)')
f.savefig('change-temp-area-best-circle-centre-y.png')

ax.clear()
ax.plot(np.diff(K_area),best_y[:diff_y.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nY Position of Centre of Best Circle (Local)')
f.savefig('change-k-area-best-circle-centre-y.png')

ax.clear()
ax.plot(np.diff(D_area),best_y[:diff_y.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nY Position of Centre of Best Circle (Local)')
f.savefig('change-d-area-best-circle-centre-y.png')

## plotting change against change y position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),diff_y,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nChange in Y Position of Centre of Best Circle (Local)')
f.savefig('change-heat-rad-change-best-circle-centre-y.png')

ax.clear()
ax.plot(np.diff(T_area),diff_y,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nChange in Y Position of Centre of Best Circle (Local)')
f.savefig('change-temp-area-change-best-circle-centre-y.png')

ax.clear()
ax.plot(np.diff(K_area),diff_y,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nChange in Y Position of Centre of Best Circle (Local)')
f.savefig('change-k-area-change-best-circle-centre-y.png')

ax.clear()
ax.plot(np.diff(D_area),diff_y,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nChange in Y Position of Centre of Best Circle (Local)')
f.savefig('change-d-area-change-best-circle-centre-y.png')

## plotting against distance from circle centre to image centre
ax.clear()
ax.plot(Qr_area,best_dist,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('heat-rad-best-circle-dist.png')

ax.clear()
ax.plot(T_area,best_dist,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('temp-area-best-circle-dist.png')

ax.clear()
ax.plot(K_area,best_dist,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('k-area-best-circle-dist.png')

ax.clear()
ax.plot(D_area,best_dist,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('d-area-best-circle-dist.png')

## plotting against change in distance from circle centre to image centre
ax.clear()
diff_dist = np.diff(best_dist)
ax.plot(Qr_area[:diff_y.shape[0]],diff_dist,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('heat-rad-change-best-circle-dist.png')

ax.clear()
ax.plot(T_area[:diff_y.shape[0]],diff_dist,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('temp-area-change-best-circle-dist.png')

ax.clear()
ax.plot(K_area[:diff_y.shape[0]],diff_dist,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('k-area-best-change-circle-dist.png')

ax.clear()
ax.plot(D_area[:diff_y.shape[0]],diff_dist,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('d-area-best-change-circle-dist.png')

## plotting change against distance from circle centre to image centre
ax.clear()
ax.plot(np.diff(Qr_area),best_dist[:diff_dist.shape[0]],'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('change-heat-rad-best-circle-dist.png')

ax.clear()
ax.plot(np.diff(T_area),best_dist[:diff_dist.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('change-temp-area-best-circle-dist.png')

ax.clear()
ax.plot(np.diff(K_area),best_dist[:diff_dist.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('change-k-area-best-circle-dist.png')

ax.clear()
ax.plot(np.diff(D_area),best_dist[:diff_dist.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nDistance from Best Circle Centre to Image Centre (Local)')
f.savefig('change-d-area-best-circle-dist.png')

## plotting change against change in distance from circle centre to image centre
ax.clear()
ax.plot(np.diff(Qr_area),diff_dist,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nChange in Distance from Best Circle Centre to Image Centre (Local)')
f.savefig('change-heat-rad-change-best-circle-dist.png')

ax.clear()
ax.plot(np.diff(T_area),diff_dist,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nChange in Distance from Best Circle Centre to Image Centre (Local)')
f.savefig('change-temp-area-change-best-circle-dist.png')

ax.clear()
ax.plot(np.diff(K_area),diff_dist,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nChange in Distance from Best Circle Centre to Image Centre (Local)')
f.savefig('change-k-area-change-best-circle-dist.png')

ax.clear()
ax.plot(np.diff(D_area),diff_dist,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nChange in Distance from Best Circle Centre to Image Centre (Local)')
f.savefig('change-d-area-change-best-circle-dist.png')

###### GLOBAL DATA #####
#//#
ax.clear()
ax.plot(best_distg)
ax.set(xlabel='Frame Index',ylabel='Distance to Centre (Pixels)')
f.suptitle('Distance Between the Best Circle Centre and Centre of Image (Global)')
f.savefig('dist-im-centre-global-best-circle-centre.png')

ax.clear()
ax.plot(best_xg)
ax.set(xlabel='Frame Index',ylabel='Best Centre, X Coordinate (Pixels)')
f.suptitle('X Coordinate of the Centre of the Best Circle (Global)')
f.savefig('best-circle-centre-x-global.png')

ax.clear()
ax.plot(best_yg)
ax.set(xlabel='Frame Index',ylabel='Best Centre, Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of the Centre of the Best Circle (Global)')
f.savefig('best-circle-centre-Y-global.png')

ax.clear()
ax.plot(np.diff(best_rg))
ax.set(xlabel='Frame Index',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Change in Best Circle Radius (Global)')
f.savefig('best-circle-radius-change-global.png')

ax.clear()
ax.plot(best_rg)
ax.set(xlabel='Frame Index',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Best Circle Radius (Global)')
f.savefig('best-circle-radius-global.png')

ax.clear()
ax.plot(best_xg,best_yg,'rx')
ax.set(xlabel='Best Circle X Position (Pixels)',ylabel='Best Circle Y Position (Pixels)',
       xlim=[0,cols],ylim=[0,rows])
f.suptitle('Coordinates of the Centres of the Best Circles (Global)')
f.savefig('best-circle-centres-global.png')

## plotting against best circle radius
ax.clear()
ax.plot(Qr_area,best_rg,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Heat Radiated and Best Circle Radius (Global)')
f.savefig('heat-rad-best-circle-radius-global.png')

ax.clear()
ax.plot(T_area,best_rg,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Area Temperature and Best Circle Radius (Global)')
f.savefig('temp-area-best-circle-radius-global.png')

ax.clear()
ax.plot(K_area,best_rg,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and Best Circle Radius (Global)')
f.savefig('k-area-best-circle-radius-global.png')

ax.clear()
ax.plot(D_area,best_rg,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and Best Circle Radius (Global)')
f.savefig('d-area-best-circle-radius-global.png')

## plotting against change in best circle radius
ax.clear()
diff_r = np.diff(best_rg)
ax.plot(Qr_area[:diff_r.shape[0]],diff_r,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Heat Radiated and \nChange in Best Circle Radius (Global)')
f.savefig('heat-rad-change-best-circle-radius-global.png')

ax.clear()
ax.plot(T_area[:diff_r.shape[0]],diff_r,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Area Temperature and \nChange in Best Circle Radius (Global)')
f.savefig('temp-area-change-best-circle-radius-global.png')

ax.clear()
ax.plot(K_area[:diff_r.shape[0]],diff_r,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\n Change in Best Circle Radius (Global)')
f.savefig('k-area-change-best-circle-radius-global.png')

ax.clear()
ax.plot(D_area[:diff_r.shape[0]],diff_r,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\n Change in Best Circle Radius (Global)')
f.savefig('d-area-change-best-circle-radius-global.png')

## plotting change against change in best circle radius
ax.clear()
ax.plot(np.diff(Qr_area),diff_r,'rx')
ax.set(xlabel='Change Heat Radiated (W)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and \n Change in Best Circle Radius (Global)')
f.savefig('change-heat-rad-change-best-circle-radius-global.png')

ax.clear()
ax.plot(np.diff(T_area),diff_r,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and \n Change in Best Circle Radius (Global)')
f.savefig('change-temp-area-change-best-circle-radius-global.png')

ax.clear()
ax.plot(np.diff(K_area),diff_r,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity \n and Change in Best Circle Radius (Global)')
f.savefig('change-k-area-change-best-circle-radius-global.png')

ax.clear()
ax.plot(np.diff(D_area),diff_r,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity \nand Change in Best Circle Radius (Global)')
f.savefig('change-d-area-change-best-circle-radius-global.png')

## plotting change against best circle radius
ax.clear()
ax.plot(np.diff(Qr_area),best_rg[:diff_r.shape[0]],'rx')
ax.set(xlabel='Change Heat Radiated (W)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and \n Change in Best Circle Radius (Global)')
f.savefig('change-heat-rad-change-best-circle-radius-global.png')

ax.clear()
ax.plot(np.diff(T_area),best_rg[:diff_r.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and \n Change in Best Circle Radius (Global)')
f.savefig('change-temp-area-change-best-circle-radius-global.png')

ax.clear()
ax.plot(np.diff(K_area),best_rg[:diff_r.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity \n and Change in Best Circle Radius (Global)')
f.savefig('change-k-area-change-best-circle-radius-global.png')

ax.clear()
ax.plot(np.diff(D_area),best_rg[:diff_r.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Best Circle Radius (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity \nand Best Circle Radius (Global)')
f.savefig('change-d-area-best-circle-radius-global.png')

## plotting against best circle score
ax.clear()
ax.plot(Qr_area,best_acg,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Heat Radiated and\n Best Circle Accumulator Score (Global)')
f.savefig('heat-rad-best-circle-accumulator-global.png')

ax.clear()
ax.plot(T_area,best_acg,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Area Temperature and\n Best Circle Accumulator Score (Global)')
f.savefig('temp-area-best-circle-accumulator-global.png')

ax.clear()
ax.plot(K_area,best_acg,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Thermal Conductivity and\n Best Circle Accumulator Score (Global)')
f.savefig('k-area-best-circle-accumulator-global.png')

ax.clear()
ax.plot(D_area,best_acg,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Thermal Diffusivity and\n Best Circle Accumulator Score (Global)')
f.savefig('d-area-best-circle-accumulator-global.png')

## plotting against change in best circle score
ax.clear()
diff_ac = np.diff(best_acg)
ax.plot(Qr_area[:diff_ac.shape[0]],diff_ac,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Heat Radiated and\n Change in Best Circle Accumulator Score (Global)')
f.savefig('heat-rad-change-best-circle-accumulator-global.png')

ax.clear()
ax.plot(T_area[:diff_ac.shape[0]],diff_ac,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Area Temperature and\n Change in Best Circle Accumulator Score (Global)')
f.savefig('temp-area-change-best-circle-accumulator-global.png')

ax.clear()
ax.plot(K_area[:diff_ac.shape[0]],diff_ac,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Thermal Conductivity and\n Change in Best Circle Accumulator Score (Global)')
f.savefig('k-area-change-best-circle-accumulator-global.png')

ax.clear()
ax.plot(D_area[:diff_ac.shape[0]],diff_ac,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Thermal Diffusivity and\n Change in Best Circle Accumulator Score (Global)')
f.savefig('d-area-change-best-circle-accumulator-global.png')

## plotting change against change in best circle score
ax.clear()
ax.plot(np.diff(Qr_area),diff_ac,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Heat Radiated and\n Change in Best Circle Accumulator Score (Global)')
f.savefig('change-heat-rad-change-best-circle-accumulator-global.png')

ax.clear()
ax.plot(np.diff(T_area),diff_ac,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Area Temperature and\n Change in Best Circle Accumulator Score (Global)')
f.savefig('change-temp-area-change-best-circle-accumulator-global.png')

ax.clear()
ax.plot(np.diff(K_area),diff_ac,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Thermal Conductivity and\n Change in Best Circle Accumulator Score (Global)')
f.savefig('change-k-area-change-best-circle-accumulator-global.png')

ax.clear()
ax.plot(np.diff(D_area),diff_ac,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Change in Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Thermal Diffusivity and\n Change in Best Circle Accumulator Score (Global)')
f.savefig('change-d-area-change-best-circle-accumulator-global.png')

## plotting change against best circle score
ax.clear()
ax.plot(np.diff(Qr_area),best_acg[:diff_ac.shape[0]],'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Heat Radiated and\nBest Circle Accumulator Score (Global)')
f.savefig('change-heat-rad-best-circle-accumulator-global.png')

ax.clear()
ax.plot(np.diff(T_area),best_acg[:diff_ac.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Area Temperature and\nBest Circle Accumulator Score (Global)')
f.savefig('change-temp-area-change-best-circle-accumulator-global.png')

ax.clear()
ax.plot(np.diff(K_area),best_acg[:diff_ac.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Thermal Conductivity and\nBest Circle Accumulator Score (Global)')
f.savefig('change-k-area-best-circle-accumulator-global.png')

ax.clear()
ax.plot(np.diff(D_area),best_acg[:diff_ac.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Best Circle Accumulator Score')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nBest Circle Accumulator Score (Global)')
f.savefig('change-d-area-best-circle-accumulator-global.png')

## plotting against x position of best circle centre
ax.clear()
ax.plot(Qr_area,best_xg,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nX Position of Centre of Best Circle (Global)')
f.savefig('heat-rad-best-circle-centre-x-global.png')

ax.clear()
ax.plot(T_area,best_xg,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nX Position of Centre of Best Circle (Global)')
f.savefig('temp-area-best-circle-centre-x-global.png')

ax.clear()
ax.plot(K_area,best_xg,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nX Position of Centre of Best Circle (Global)')
f.savefig('k-area-best-circle-centre-x-global.png')

ax.clear()
ax.plot(D_area,best_xg,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nX Position of Centre of Best Circle (Global)')
f.savefig('d-area-best-circle-centre-x-global.png')

## plotting against change x position of best circle centre
ax.clear()
diff_x = np.diff(best_xg)
ax.plot(Qr_area[:diff_x.shape[0]],diff_x,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('heat-rad-change-best-circle-centre-x-global.png')

ax.clear()
ax.plot(T_area[:diff_x.shape[0]],diff_x,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('temp-area-change-best-circle-centre-x-global.png')

ax.clear()
ax.plot(K_area[:diff_x.shape[0]],diff_x,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('k-area-best-change-circle-centre-x-global.png')

ax.clear()
ax.plot(D_area[:diff_x.shape[0]],diff_x,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('d-area-best-change-circle-centre-x-global.png')

## plotting change against x position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),best_xg[:diff_x.shape[0]],'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nX Position of Centre of Best Circle (Global)')
f.savefig('change-heat-rad-best-circle-centre-x-global.png')

ax.clear()
ax.plot(np.diff(T_area),best_xg[:diff_x.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nX Position of Centre of Best Circle (Global)')
f.savefig('change-temp-area-best-circle-centre-x-global.png')

ax.clear()
ax.plot(np.diff(K_area),best_xg[:diff_x.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nX Position of Centre of Best Circle (Global)')
f.savefig('change-k-area-best-circle-centre-x-global.png')

ax.clear()
ax.plot(np.diff(D_area),best_xg[:diff_x.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nX Position of Centre of Best Circle (Global)')
f.savefig('change-d-area-best-circle-centre-x-global.png')

## plotting change agsinst chhange in x position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),diff_x,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('change-heat-rad-change-best-circle-centre-x-global.png')

ax.clear()
ax.plot(np.diff(T_area),diff_x,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('change-temp-area-change-best-circle-centre-x-global.png')

ax.clear()
ax.plot(np.diff(K_area),diff_x,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('change-k-area-change-best-circle-centre-x-global.png')

ax.clear()
ax.plot(np.diff(D_area),diff_x,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('change-d-area-change-best-circle-centre-x-global.png')

## plotting change against change x position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),diff_x,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('change-heat-rad-change-best-circle-centre-x-global.png')

ax.clear()
ax.plot(np.diff(T_area),diff_x,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('change-temp-area-change-best-circle-centre-x-global.png')

ax.clear()
ax.plot(np.diff(K_area),diff_x,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('change-k-area-change-best-circle-centre-x-global.png')

ax.clear()
ax.plot(np.diff(D_area),diff_x,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='X Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nChange in X Position of Centre of Best Circle (Global)')
f.savefig('change-d-area-change-best-circle-centre-x-global.png')

## plotting against y position of best circle centre
ax.clear()
ax.plot(Qr_area,best_yg,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nY Position of Centre of Best Circle (Global)')
f.savefig('heat-rad-best-circle-centre-y-global.png')

ax.clear()
ax.plot(T_area,best_yg,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nY Position of Centre of Best Circle (Global)')
f.savefig('temp-area-best-circle-centre-y-global.png')

ax.clear()
ax.plot(K_area,best_yg,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nY Position of Centre of Best Circle (Global)')
f.savefig('k-area-best-circle-centre-y-global.png')

ax.clear()
ax.plot(D_area,best_yg,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nY Position of Centre of Best Circle (Global)')
f.savefig('d-area-best-circle-centre-y-global.png')

## plotting against change y position of best circle centre
ax.clear()
diff_y = np.diff(best_yg)
ax.plot(Qr_area[:diff_y.shape[0]],diff_y,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nChange in Y Position of Centre of Best Circle (Global)')
f.savefig('heat-rad-change-best-circle-centre-y-global.png')

ax.clear()
ax.plot(T_area[:diff_y.shape[0]],diff_y,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nChange in Y Position of Centre of Best Circle (Global)')
f.savefig('temp-area-change-best-circle-centre-y-global.png')

ax.clear()
ax.plot(K_area[:diff_y.shape[0]],diff_y,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nChange in Y Position of Centre of Best Circle (Global)')
f.savefig('k-area-best-change-circle-centre-y-global.png')

ax.clear()
ax.plot(D_area[:diff_y.shape[0]],diff_y,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nChange in Y Position of Centre of Best Circle (Global)')
f.savefig('d-area-best-change-circle-centre-y-global.png')

## plotting change against y position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),best_yg[:diff_y.shape[0]],'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nY Position of Centre of Best Circle (Global)')
f.savefig('change-heat-rad-best-circle-centre-y-global.png')

ax.clear()
ax.plot(np.diff(T_area),best_yg[:diff_y.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nX Position of Centre of Best Circle (Global)')
f.savefig('change-temp-area-best-circle-centre-y-global.png')

ax.clear()
ax.plot(np.diff(K_area),best_yg[:diff_y.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nY Position of Centre of Best Circle (Global)')
f.savefig('change-k-area-best-circle-centre-y-global.png')

ax.clear()
ax.plot(np.diff(D_area),best_yg[:diff_y.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nY Position of Centre of Best Circle (Global)')
f.savefig('change-d-area-best-circle-centre-y-global.png')

## plotting change against change y position of best circle centre
ax.clear()
ax.plot(np.diff(Qr_area),diff_y,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nChange in Y Position of Centre of Best Circle (Global)')
f.savefig('change-heat-rad-change-best-circle-centre-y-global.png')

ax.clear()
ax.plot(np.diff(T_area),diff_y,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nChange in Y Position of Centre of Best Circle (Global)')
f.savefig('change-temp-area-change-best-circle-centre-y-global.png')

ax.clear()
ax.plot(np.diff(K_area),diff_y,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nChange in Y Position of Centre of Best Circle (Global)')
f.savefig('change-k-area-change-best-circle-centre-y-global.png')

ax.clear()
ax.plot(np.diff(D_area),diff_y,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Y Position of Best Circle Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nChange in Y Position of Centre of Best Circle (Global)')
f.savefig('change-d-area-change-best-circle-centre-y-global.png')

## plotting against distance from circle centre to image centre
ax.clear()
ax.plot(Qr_area,best_dist,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('heat-rad-best-circle-dist-global.png')

ax.clear()
ax.plot(T_area,best_dist,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('temp-area-best-circle-dist-global.png')

ax.clear()
ax.plot(K_area,best_dist,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('k-area-best-circle-dist-global.png')

ax.clear()
ax.plot(D_area,best_dist,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('d-area-best-circle-dist-global.png')

## plotting against change in distance from circle centre to image centre
ax.clear()
diff_dist = np.diff(best_distg)
ax.plot(Qr_area[:diff_y.shape[0]],diff_dist,'rx')
ax.set(xlabel='Heat Radiated (W)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Heat Radiated and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('heat-rad-change-best-circle-dist-global.png')

ax.clear()
ax.plot(T_area[:diff_y.shape[0]],diff_dist,'rx')
ax.set(xlabel='Area Temperature (K)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Area Temperature and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('temp-area-change-best-circle-dist-global.png')

ax.clear()
ax.plot(K_area[:diff_y.shape[0]],diff_dist,'rx')
ax.set(xlabel='Thermal Conductivity (W.mK)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Thermal Conductivity and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('k-area-best-change-circle-dist-global.png')

ax.clear()
ax.plot(D_area[:diff_y.shape[0]],diff_dist,'rx')
ax.set(xlabel='Thermal Diffusivity (m2/s)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Thermal Diffusivity and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('d-area-best-change-circle-dist-global.png')

## plotting change against distance from circle centre to image centre
ax.clear()
ax.plot(np.diff(Qr_area),best_distg[:diff_dist.shape[0]],'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('change-heat-rad-best-circle-dist-global.png')

ax.clear()
ax.plot(np.diff(T_area),best_distg[:diff_dist.shape[0]],'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('change-temp-area-best-circle-dist-global.png')

ax.clear()
ax.plot(np.diff(K_area),best_distg[:diff_dist.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('change-k-area-best-circle-dist-global.png')

ax.clear()
ax.plot(np.diff(D_area),best_distg[:diff_dist.shape[0]],'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nDistance from Best Circle Centre to Image Centre (Global)')
f.savefig('change-d-area-best-circle-dist-global.png')

## plotting change against change in distance from circle centre to image centre
ax.clear()
ax.plot(np.diff(Qr_area),diff_dist,'rx')
ax.set(xlabel='Change in Heat Radiated (W)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Heat Radiated and\nChange in Distance from Best Circle Centre to Image Centre (Global)')
f.savefig('change-heat-rad-change-best-circle-dist-global.png')

ax.clear()
ax.plot(np.diff(T_area),diff_dist,'rx')
ax.set(xlabel='Change in Area Temperature (K)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Area Temperature and\nChange in Distance from Best Circle Centre to Image Centre (Global)')
f.savefig('change-temp-area-change-best-circle-dist-global.png')

ax.clear()
ax.plot(np.diff(K_area),diff_dist,'rx')
ax.set(xlabel='Change in Thermal Conductivity (W.mK)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Conductivity and\nChange in Distance from Best Circle Centre to Image Centre (Global)')
f.savefig('change-k-area-change-best-circle-dist-global.png')

ax.clear()
ax.plot(np.diff(D_area),diff_dist,'rx')
ax.set(xlabel='Change in Thermal Diffusivity (m2/s)',ylabel='Change in Distance from Image Centre (Pixels)')
f.suptitle('Relationship between Change in Thermal Diffusivity and\nChange in Distance from Best Circle Centre to Image Centre (Global)')
f.savefig('change-d-area-change-best-circle-dist-global.png')
#//#

## plotting just the peak values
# tolerance when searching for values at or near peak
# e.g. 0.005 equals tolerance of 0.5%
tol = 0.005
# get peak values for set
I_max = I.max(axis=(0,1))
# find where values are close to peak
i = np.where(I_max>=(1.0-tol)*I_max.max())[0]

ax.clear()
ax.plot(best_x,best_y,'bx',best_x[i],best_y[i],'rx')
ax.legend(['All','Peak'])
f.suptitle('Location of the Best Circle Centre with Marked Values Showing\n when the Power Density was Near Peak (Local)')
f.savefig('marked-best-circle-centre.png')

ax.clear()
ax.plot(best_xg,best_yg,'bx',best_xg[i],best_yg[i],'rx')
ax.legend(['All','Peak'])
f.suptitle('Location of the Best Circle Centre with Marked Values Showing\n when the Power Density was Near Peak (Global)')
f.savefig('marked-best-circle-centre-global.png')

ax.clear()
ax.plot(best_r,'bx',i,best_r[i],'rx')
ax.legend(['All','Peak'])
f.suptitle('Location of the Best Circle Centre with Marked Values Showing\n when the Power Density was Near Peak (Local)')
f.savefig('marked-best-circle-radius.png')

ax.clear()
ax.plot(best_rg,'bx',i,best_rg[i],'rx')
ax.legend(['All','Peak'])
f.suptitle('Best Circle Radius with Marked Values Showing\n when the Power Density was Near Peak (Global)')
f.savefig('marked-best-circle-radius-global.png')


## write the data to a file
# local
local_data =  zip_longest(Qr_area,K_area,D_area,T_area,best_r,best_ac,best_x,best_y,best_dist)
# global
global_data = zip_longest(Qr_area,K_area,D_area,T_area,best_rg,best_acg,best_xg,best_yg,best_distg)

headers =['Radiated Heat Times Area (W)','Thermal Conductivity of Area (W.mK)','Thermal Diffusivity of Area (m2/s)','Temperature of Area (K)',
                              'Best Circle Radius (Pixels)','Best Circle Accumulator Score','Best Circle X Position (Pixels)',
                              'Best Circle Y Position (Pixels)','Best Circle Distance from Centre (Pixels)']

with open('thermal-hough-circle-metrics-local.csv.',mode='w') as file:
   writer = csv.writer(file)
   writer.writerow(headers)
   writer.writerows(local_data)
   
with open('thermal-hough-circle-metrics-global.csv.',mode='w') as file:
   writer = csv.writer(file)
   writer.writerow(headers)
   writer.writerows(global_data)
   


