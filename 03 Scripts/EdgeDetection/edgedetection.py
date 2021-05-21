from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.signal import find_peaks
from scipy.fftpack import fftn,ifftn,fftshift
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from skimage.filters import prewitt_v,prewitt_h,prewitt
from skimage.transform import hough_circle, hough_circle_peaks
import os
import cv2
#import cmapy

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

def powerEstBestCircle(I,radii_range,pixel_pitch):
   # normalize image so it can be used by skimage
   I_norm = I/I.max(axis=(0,1))
   # search for circles
   res = hough_circle(I_norm,radii_range)
   # get the top three circles
   accums,cx,cy,radii = hough_circle_peaks(res,radii_range,total_num_peaks=1)
   # choose the highest rated circle to estimate power with
   return powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I,pixel_pitch)

def powerEstAccumCircle(I,radii_range,PP,min_accum=0.6,num_peaks=10):
   # normalize image so it can be used by skimage
   I_norm = I/I.max(axis=(0,1))
   # search for circles
   res = hough_circle(I_norm,radii_range)
   # get the top three circles
   accums,cx,cy,radii = hough_circle_peaks(res,radii_range,total_num_peaks=num_peaks)
   # search for any circles with the min score
   ai = np.where(accums>=min_accum)[0]
   # if there are none, then return zero
   if ai.shape[0]==0:
      return 0.0
   # else choose the highest scoring one
   else:
      # as accums are already sorted by value, the highest score in the filtered
      # list is the first one.
      return powerEstHoughCircle([cx[ai][0],cy[ai][0],radii[ai][0]],I,pixel_pitch)

def collectHoughStats(accums,cx,cy,radii):
   return accums.max(),cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]
# path to footage
path = "D:/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"
print("Reading in video data")
frames_set_array = readH264(path)
rows,cols,depth = frames_set_array.shape

target_f = depth # 1049
pixel_pitch=20.0e-6
# sampling frequency,radiiof thermal camera, assumed
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

print("Making directories")
# raw difference matricies
os.makedirs("DiffMatricies",exist_ok=True)
# Sobel operator, gaussian + gradient
os.makedirs("SobelXResults",exist_ok=True)
os.makedirs("SobelYResults",exist_ok=True)
os.makedirs("SobelSumResults",exist_ok=True)
# Laplace operator
os.makedirs("LaplacianResults",exist_ok=True)
# Scharr, eq to Sobel with 3x3 kernel
os.makedirs("ScharrXResults",exist_ok=True)
os.makedirs("ScharrYResults",exist_ok=True)
os.makedirs("ScharrSumResults",exist_ok=True)
# Prewitt 
os.makedirs("PrewittVerticalResults",exist_ok=True)
os.makedirs("PrewittHorizResults",exist_ok=True)
os.makedirs("PrewittEdgeResults",exist_ok=True)

print("Normalizing values")
I_norm_cv = ((I/I.max(axis=(0,1)).max())*255).astype('uint8')

print("Starting processing")
##print("Generating edge results")
##for ff in range(I.shape[2]):
##    # raw difference matrix
##    cv2.imwrite("DiffMatricies/diff-f{0}.png".format(ff),diffRow(I_norm_cv[:,:,ff])+diffCol(I_norm_cv[:,:,ff]))
##    # Sobel operation in x and y direction
##    sobel_x = cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,1,0,ksize=5)
##    sobel_y = cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,1,0,ksize=5)
##    cv2.imwrite("SobelXResults/sobel-x-f{0}.png".format(ff),sobel_x)
##    cv2.imwrite("SobelYResults/sobel-Y-f{0}.png".format(ff),sobel_y)
##    cv2.imwrite("SobelSumResults/sobel-sum-f{0}.png".format(ff),np.abs(sobel_x)+np.abs(sobel_y))
##    # Scharr operation
##    scharr_x = cv2.Scharr(I_norm_cv[:,:,ff],cv2.CV_8U,1,0)
##    scharr_y = cv2.Scharr(I_norm_cv[:,:,ff],cv2.CV_8U,0,1)
##    cv2.imwrite("ScharrXResults/scharr-x-f{0}.png".format(ff),scharr_x)
##    cv2.imwrite("ScharrYResults/scharr-Y-f{0}.png".format(ff),scharr_y)
##    cv2.imwrite("ScharrSumResults/scharr-sum-f{0}.png".format(ff),np.abs(scharr_x)+np.abs(scharr_y))
##    # Laplacian operation, second derivative
##    laplace = cv2.Laplacian(I_norm_cv[:,:,ff],cv2.CV_8U,ksize=3)
##    cv2.imwrite("LaplacianResults/laplace-f{0}.png".format(ff),laplace)
##    # Prewitt results
##    # results are [-1,1] so need to be scaled to [0,255] so it can be saved as an image
##    cv2.imwrite("PrewittVerticalResults/prewitt-v-f{0}.png".format(ff),np.interp(prewitt_v(I_norm_cv[:,:,ff]),[-1.0,1.0],[0,255]).astype('uint8'))
##    cv2.imwrite("PrewittHorizResults/prewitt-h-f{0}.png".format(ff),np.interp(prewitt_h(I_norm_cv[:,:,ff]),[-1.0,1.0],[0,255]).astype('uint8'))
##    cv2.imwrite("PrewittEdgeResults/prewitt-edge-f{0}.png".format(ff),np.interp(prewitt(I_norm_cv[:,:,ff]),[-1.0,1.0],[0,255]).astype('uint8'))

radii_range = np.arange(30,64,1)
##pest_diff = np.zeros(I.shape[2])
##pest_sobel_x = np.zeros(I.shape[2])
##pest_sobel_y = np.zeros(I.shape[2])
##pest_sobel_sum = np.zeros(I.shape[2])
##pest_scharr_x = np.zeros(I.shape[2])
##pest_scharr_y = np.zeros(I.shape[2])
##pest_scharr_sum = np.zeros(I.shape[2])
##pest_laplace = np.zeros(I.shape[2])
##pest_prewitt_v = np.zeros(I.shape[2])
##pest_prewitt_h = np.zeros(I.shape[2])
##pest_prewitt = np.zeros(I.shape[2])
##print("Starting power estimate")
##for ff in range(I.shape[2]):
##   print(ff)
##   ## difference matrix
##   res = hough_circle(diffRow(I_norm_cv[:,:,ff])+diffCol(I_norm_cv[:,:,ff]),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_diff[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   ## sobel
##   res = hough_circle(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,1,0,ksize=5),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_sobel_x[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   res = hough_circle(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,0,1,ksize=5),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_sobel_y[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   res = hough_circle(np.abs(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,1,0,ksize=5))+np.abs(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,0,1,ksize=5)),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_sobel_sum[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   ## scharr
##   res = hough_circle(cv2.Scharr(I_norm_cv[:,:,ff],cv2.CV_8U,1,0),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_scharr_x[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   res = hough_circle(cv2.Scharr(I_norm_cv[:,:,ff],cv2.CV_8U,0,1),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_scharr_y[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   res = hough_circle(np.abs(cv2.Scharr(I_norm_cv[:,:,ff],cv2.CV_8U,1,0))+np.abs(cv2.Scharr(I_norm_cv[:,:,ff],cv2.CV_8U,0,1)),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_scharr_sum[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   ## laplacian
##   res = hough_circle(cv2.Laplacian(I_norm_cv[:,:,ff],cv2.CV_8U,ksize=3),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_laplace[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   ## prewitt
##   res = hough_circle(np.interp(prewitt_v(I_norm_cv[:,:,ff]),[-1.0,1.0],[0,255]).astype('uint8'),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_prewitt_v[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   res = hough_circle(np.interp(prewitt_h(I_norm_cv[:,:,ff]),[-1.0,1.0],[0,255]).astype('uint8'),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_prewitt_h[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##   res = hough_circle(np.interp(prewitt(I_norm_cv[:,:,ff]),[-1.0,1.0],[0,255]).astype('uint8'),radii_range)
##   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
##   pest_prewitt[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
##
#### plot power results
f,ax = plt.subplots()
##ax.plot(pest_diff)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Raw Difference Matricies')
##f.savefig('pest-diff-matricies.png')
##
##ax.clear()
##ax.plot(pest_sobel_x)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Sobel Edge Detection, X')
##f.savefig('pest-sobel-x.png')
##
##ax.clear()
##ax.plot(pest_sobel_y)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Sobel Edge Detection, Y')
##f.savefig('pest-sobel-y.png')
##
##ax.clear()
##ax.plot(pest_sobel_sum)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Sobel Edge Detection, Sum')
##f.savefig('pest-sobel-sum.png')
##
##ax.clear()
##ax.plot(pest_scharr_x)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Scharr Edge Detection, X')
##f.savefig('pest-scharr-x.png')
##
##ax.clear()
##ax.plot(pest_scharr_y)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Scharr Edge Detection, Y')
##f.savefig('pest-scharr-y.png')
##
##ax.clear()
##ax.plot(pest_scharr_sum)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Scharr Edge Detection, Sum')
##f.savefig('pest-scharr-sum.png')
##
##ax.clear()
##ax.plot(pest_laplace)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Laplace Edge Detection')
##f.savefig('pest-laplace.png')
##
##ax.clear()
##ax.plot(pest_prewitt_v)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Scharr Edge Detection, Y')
##f.savefig('pest-prewitt-v.png')
##
##ax.clear()
##ax.plot(pest_prewitt_h)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Scharr Edge Detection, X')
##f.savefig('pest-prewitt-h.png')
##
##ax.clear()
##ax.plot(pest_prewitt)
##ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
##f.suptitle('Power Estimate Using Results of Scharr Edge Detection, Sum')
##f.savefig('pest-prewitt-sum.png')

os.makedirs("ScharrSumDenoise\Single",exist_ok=True)
os.makedirs("ScharrSumDenoise\Multi",exist_ok=True)
os.makedirs("SobelSumDenoise\Single",exist_ok=True)
os.makedirs("SobelSumDenoise\Multi",exist_ok=True)
# power estimates according to the different denoising methods
pest_sobel_single = np.zeros(I.shape[2])
pest_sobel_multi = np.zeros(I.shape[2])
pest_scharr_single = np.zeros(I.shape[2])
pest_scharr_multi = np.zeros(I.shape[2])
# power estimate before denoising
pest_scharr_raw = np.zeros(I.shape[2])
pest_sobel_raw = np.zeros(I.shape[2])

## hough circle data
# matricies are reused
# raw
hough_accum_scharr = np.zeros(I.shape[2])
hough_x_scharr = np.zeros(I.shape[2])
hough_y_scharr = np.zeros(I.shape[2])
hough_r_scharr = np.zeros(I.shape[2])
hough_accum_sobel = np.zeros(I.shape[2])
hough_x_sobel = np.zeros(I.shape[2])
hough_y_sobel = np.zeros(I.shape[2])
hough_r_sobel = np.zeros(I.shape[2])
# denoised
hough_accumd_scharr = np.zeros(I.shape[2])
hough_xd_scharr = np.zeros(I.shape[2])
hough_yd_scharr = np.zeros(I.shape[2])
hough_rd_scharr = np.zeros(I.shape[2])
hough_accumd_sobel = np.zeros(I.shape[2])
hough_xd_sobel = np.zeros(I.shape[2])
hough_yd_sobel = np.zeros(I.shape[2])
hough_rd_sobel = np.zeros(I.shape[2])

hough_accumd_scharr_multi = np.zeros(I.shape[2])
hough_xd_scharr_multi = np.zeros(I.shape[2])
hough_yd_scharr_multi = np.zeros(I.shape[2])
hough_rd_scharr_multi = np.zeros(I.shape[2])
hough_accumd_sobel_multi = np.zeros(I.shape[2])
hough_xd_sobel_multi = np.zeros(I.shape[2])
hough_yd_sobel_multi = np.zeros(I.shape[2])
hough_rd_sobel_multi = np.zeros(I.shape[2])

# results frame list
img_list = []
print("Starting power estimate Sobel (denoising)")
for ff in range(I.shape[2]):
   #print("\rSobel Single {}".format(ff),end='')
   ## sobel
   sobel_res = np.abs(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,1,0,ksize=5))+np.abs(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,0,1,ksize=5))
   res = hough_circle(sobel_res,radii_range)
   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
   pest_sobel_raw[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
   hough_accum_sobel[ff],hough_x_sobel[ff],hough_y_sobel[ff],hough_r_sobel[ff] = collectHoughStats(accums,cx,cy,radii)
   
   # apply denoising
   sobel_res_d = cv2.fastNlMeansDenoising(sobel_res,dst=None,templateWindowSize=10,searchWindowSize=21,h=40.0)
   #cv2.imwrite(r'SobelSumDenoise\Single\frame-denoise-single-{}.png'.format(ff),sobel_res_d)
   res = hough_circle(sobel_res_d,radii_range)
   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
   pest_sobel_single[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
   hough_accumd_sobel[ff],hough_xd_sobel[ff],hough_yd_sobel[ff],hough_rd_sobel[ff] = collectHoughStats(accums,cx,cy,radii)
   # add image to list to be used in the next loop
   img_list.append(sobel_res)

for ff in range(1,len(img_list)):
   #print("\rSobel Multi {}".format(ff),end='')
   sobel_res_d = cv2.fastNlMeansDenoisingMulti(img_list,ff,1,None,4,7,35)
   #cv2.imwrite("SobelSumDenoise\Multi\frame-denoise-multi-{}.png".format(ff),sobel_res_d)
   res = hough_circle(sobel_res_d,radii_range)
   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
   pest_sobel_multi[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
   hough_accumd_sobel_multi[ff],hough_xd_sobel_multi[ff],hough_yd_sobel_multi[ff],hough_rd_sobel_multi[ff] = collectHoughStats(accums,cx,cy,radii)
   
print("Starting power estimate Scharr (denoising)")
img_list = []
for ff in range(I.shape[2]):
   print("\rScharr Single {}".format(ff),end='')
   ## scharr
   scharr_res = np.abs(cv2.Scharr(I_norm_cv[:,:,ff],cv2.CV_8U,1,0))+np.abs(cv2.Scharr(I_norm_cv[:,:,ff],cv2.CV_8U,0,1))
   res = hough_circle(scharr_res,radii_range)
   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
   pest_scharr_raw[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
   hough_accum_scharr[ff],hough_x_scharr[ff],hough_y_scharr[ff],hough_r_scharr[ff] = collectHoughStats(accums,cx,cy,radii)
   
   scharr_res_d = cv2.fastNlMeansDenoising(scharr_res,dst=None,templateWindowSize=10,searchWindowSize=21,h=40.0)
   #cv2.imwrite(r'ScharrSumDenoise\Single\frame-denoise-single-{}.png'.format(ff),scharr_res_d)
   res = hough_circle(scharr_res_d,radii_range)
   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
   pest_scharr_single[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
   hough_accumd_scharr[ff],hough_xd_scharr[ff],hough_yd_scharr[ff],hough_rd_scharr[ff] = collectHoughStats(accums,cx,cy,radii)
   img_list.append(scharr_res)

for ff in range(1,len(img_list)):
   print("\rScharr Multi {}".format(ff),end='')
   scharr_res_d = cv2.fastNlMeansDenoisingMulti(img_list,ff,1,None,4,7,35)
   #cv2.imwrite("ScharrSumDenoise\Multi\frame-denoise-multi-{}.png".format(ff),scharr_res_d)
   res = hough_circle(scharr_res_d,radii_range)
   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
   pest_scharr_multi[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
   hough_accumd_scharr_multi[ff],hough_xd_scharr_multi[ff],hough_yd_scharr_multi[ff],hough_rd_scharr_multi[ff] = collectHoughStats(accums,cx,cy,radii)

print("Plotting power estimates")
ax.clear()
ax.plot(pest_scharr_single)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using Single Denoised Results of \n Scharr Edge Detection, Sum')
f.savefig('pest-scharr-sum-denoise-single.png')

ax.clear()
ax.plot(pest_scharr_multi)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using Multi Denoised Results of \n Scharr Edge Detection, Sum')
f.savefig('pest-scharr-sum-denoise-multi.png')

ax.clear()
ax.plot(pest_sobel_single)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using Single Denoised Results of \n Sobel Edge Detection, Sum')
f.savefig('pest-sobel-sum-denoise-single.png')

ax.clear()
ax.plot(pest_sobel_multi)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using Multi Denoised Results of \n Sobel Edge Detection, Sum')
f.savefig('pest-sobel-sum-denoise-multi.png')

print("Plotting Hough Circle data")
ax.clear()
ax.plot(hough_accum_scharr)
ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
f.suptitle('Highest Hough Circle Accumulator Score found after Applying Scharr Edge Detection')
f.savefig('scharr-highest-hough-accum.png'.format(file_name))

ax.clear()
ax.plot(hough_accumd_scharr)
ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
f.suptitle('Highest Hough Circle Accumulator Score after using Single Non-Linear Means Denoising\n on the Results of Scharr Edge Detection')
f.savefig('scharr-highest-filt-hough-accum.png'.format(file_name))

ax.clear()
ax.plot(hough_accumd_scharr_multi)
ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
f.suptitle('Highest Hough Circle Accumulator Score after using Multi Non-Linear Means Denoising\n on the Results of Scharr Edge Detection')
f.savefig('scharr-highest-filt-multi-hough-accum.png'.format(file_name))

ax.clear()
ax.plot(hough_x_scharr)
ax.set(xlabel='Frame Index',ylabel='X Coordinate (Pixels)')
f.suptitle('X Coordinate of Highest Scoring Hough Circle found after Applying Scharr Edge Detection')
f.savefig('scharr-hough-best-circle-centre-x.png'.format(file_name))

ax.clear()
ax.plot(hough_xd_scharr)
ax.set(xlabel='Frame Index',ylabel='X Coordinate (Pixels)')
f.suptitle('X Coordinate of Highest Scoring Hough Circle after using Single Non-Linear Means Denoising\n on the Results of Scharr Edge Detection')
f.savefig('scharr-filt-hough-best-circle-centre-x.png'.format(file_name))

ax.clear()
ax.plot(hough_xd_scharr_multi)
ax.set(xlabel='Frame Index',ylabel='X Coordinate (Pixels)')
f.suptitle('X Coordinate of Highest Scoring Hough Circle after using Multi Non-Linear Means Denoising\n on the Results of Scharr Edge Detection')
f.savefig('scharr-filt-multi-hough-best-circle-centre-x.png'.format(file_name))

ax.clear()
ax.plot(hough_y_scharr)
ax.set(xlabel='Frame Index',ylabel='Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of Highest Scoring Hough Circle after Applying Scharr Edge Detection')
f.savefig('scharr-hough-best-circle-centre-y.png'.format(file_name))

ax.clear()
ax.plot(hough_yd_scharr)
ax.set(xlabel='Frame Index',ylabel='Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of Highest Scoring Hough Circle after using Single Non-Linear Means Denoising\n on the Results of Scharr Edge Detection')
f.savefig('scharr-filt-hough-best-circle-centre-y.png'.format(file_name))

ax.clear()
ax.plot(hough_yd_scharr_multi)
ax.set(xlabel='Frame Index',ylabel='Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of Highest Scoring Hough Circle after using Multi Non-Linear Means Denoising\n on the Results of Scharr Edge Detection')
f.savefig('scharr-filt-multi-hough-best-circle-centre-y.png'.format(file_name))

ax.clear()
ax.plot(hough_r_scharr)
ax.set(xlabel='Frame Index',ylabel='Circle Radius (Pixels)')
f.suptitle('Radius of the Highest Scoring Hough Circle after Applying Scharr Edge Detection')
f.savefig('scharr-hough-best-circle-radius.png')

ax.clear()
ax.plot(hough_rd_scharr)
ax.set(xlabel='Frame Index',ylabel='Circle Radius (Pixels)')
f.suptitle('Radius of the Highest Scoring Hough Circle after using Single Non-Linear Means Denoising\n on the Results of Scharr Edge Detection')
f.savefig('scharr-filt-hough-best-circle-radius.png')

ax.clear()
ax.plot(hough_rd_scharr_multi)
ax.set(xlabel='Frame Index',ylabel='Circle Radius (Pixels)')
f.suptitle('Radius of the Highest Scoring Hough Circle after using Multi Non-Linear Means Denoising\n on the Results of Scharr Edge Detection')
f.savefig('scharr-filt-multi-hough-best-circle-radius.png')

## sobel
ax.clear()
ax.plot(hough_accum_sobel)
ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
f.suptitle('Highest Hough Circle Accumulator Score found after Applying Sobel Edge Detection')
f.savefig('sobel-highest-hough-accum.png'.format(file_name))

ax.clear()
ax.plot(hough_accumd_sobel)
ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
f.suptitle('Highest Hough Circle Accumulator Score after using Single Non-Linear Means Denoising\n on the Results of Sobel Edge Detection')
f.savefig('sobel-highest-filt-hough-accum.png'.format(file_name))

ax.clear()
ax.plot(hough_accumd_sobel_multi)
ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
f.suptitle('Highest Hough Circle Accumulator Score after using Multi Non-Linear Means Denoising\n on the Results of Sobel Edge Detection')
f.savefig('sobel-highest-filt-multi-hough-accum.png'.format(file_name))

ax.clear()
ax.plot(hough_x_sobel)
ax.set(xlabel='Frame Index',ylabel='X Coordinate (Pixels)')
f.suptitle('X Coordinate of Highest Scoring Hough Circle found after Applying Sobel Edge Detection')
f.savefig('sobel-hough-best-circle-centre-x.png'.format(file_name))

ax.clear()
ax.plot(hough_xd_sobel)
ax.set(xlabel='Frame Index',ylabel='X Coordinate (Pixels)')
f.suptitle('X Coordinate of Highest Scoring Hough Circle after using Single Non-Linear Means Denoising\n on the Results of Sobel Edge Detection')
f.savefig('sobel-filt-hough-best-circle-centre-x.png'.format(file_name))

ax.clear()
ax.plot(hough_xd_sobel_multi)
ax.set(xlabel='Frame Index',ylabel='X Coordinate (Pixels)')
f.suptitle('X Coordinate of Highest Scoring Hough Circle after using Multi Non-Linear Means Denoising\n on the Results of Sobel Edge Detection')
f.savefig('sobel-filt-multi-hough-best-circle-centre-x.png'.format(file_name))

ax.clear()
ax.plot(hough_y_sobel)
ax.set(xlabel='Frame Index',ylabel='Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of Highest Scoring Hough Circle after Applying Sobel Edge Detection')
f.savefig('sobel-hough-best-circle-centre-y.png'.format(file_name))

ax.clear()
ax.plot(hough_yd_sobel)
ax.set(xlabel='Frame Index',ylabel='Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of Highest Scoring Hough Circle after using Single Non-Linear Means Denoising\n on the Results of Sobel Edge Detection')
f.savefig('sobel-filt-hough-best-circle-centre-y.png'.format(file_name))

ax.clear()
ax.plot(hough_yd_sobel_multi)
ax.set(xlabel='Frame Index',ylabel='Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of Highest Scoring Hough Circle after using Multi Non-Linear Means Denoising\n on the Results of Sobel Edge Detection')
f.savefig('sobel-filt-multi-hough-best-circle-centre-y.png'.format(file_name))

ax.clear()
ax.plot(hough_r_sobel)
ax.set(xlabel='Frame Index',ylabel='Circle Radius (Pixels)')
f.suptitle('Radius of the Highest Scoring Hough Circle after Applying Sobel Edge Detection')
f.savefig('sobel-hough-best-circle-radius.png')

ax.clear()
ax.plot(hough_rd_sobel)
ax.set(xlabel='Frame Index',ylabel='Circle Radius (Pixels)')
f.suptitle('Radius of the Highest Scoring Hough Circle after using Single Non-Linear Means Denoising\n on the Results of Sobel Edge Detection')
f.savefig('sobel-filt-hough-best-circle-radius.png')

ax.clear()
ax.plot(hough_rd_sobel_multi)
ax.set(xlabel='Frame Index',ylabel='Circle Radius (Pixels)')
f.suptitle('Radius of the Highest Scoring Hough Circle after using Multi Non-Linear Means Denoising\n on the Results of Sobel Edge Detection')
f.savefig('sobel-filt-multi-hough-best-circle-radius.png')

print("Saving power results to file")
np.savetxt("pest-scharr-sum-raw.csv",pest_scharr_raw,delimiter=',')
np.savetxt("pest-sobel-sum-raw.csv",pest_sobel_raw,delimiter=',')
np.savetxt("pest-scharr-sum-denoise-single.csv",pest_scharr_single,delimiter=',')
np.savetxt("pest-scharr-sum-denoise-multi.csv",pest_scharr_multi,delimiter=',')
np.savetxt("pest-sobel-sum-denoise-single.csv",pest_sobel_single,delimiter=',')
np.savetxt("pest-sobel-sum-denoise-multi.csv",pest_sobel_multi,delimiter=',')
print("Saving Hough Circle stats to file")
# raw
np.savetxt("hough-circle-accum-scharr-raw.csv",hough_accum_scharr,delimiter=',')
np.savetxt("hough-circle-centre-x-scharr-sum-raw.csv",hough_x_scharr,delimiter=',')
np.savetxt("hough-circle-centre-y-scharr-sum-raw.csv",hough_y_scharr,delimiter=',')
np.savetxt("hough-circle-centre-r-scharr-sum-raw.csv",hough_r_scharr,delimiter=',')

np.savetxt("hough-circle-accum-sobel-raw.csv",hough_accum_sobel,delimiter=',')
np.savetxt("hough-circle-centre-x-sobel-sum-raw.csv",hough_x_sobel,delimiter=',')
np.savetxt("hough-circle-centre-y-sobel-sum-raw.csv",hough_y_sobel,delimiter=',')
np.savetxt("hough-circle-centre-r-sobel-sum-raw.csv",hough_r_sobel,delimiter=',')
# single denoise
np.savetxt("hough-circle-accum-scharr-nl-single.csv",hough_accumd_scharr,delimiter=',')
np.savetxt("hough-circle-centre-x-scharr-sum-nl-single.csv",hough_xd_scharr,delimiter=',')
np.savetxt("hough-circle-centre-y-scharr-sum-nl-single.csv",hough_yd_scharr,delimiter=',')
np.savetxt("hough-circle-centre-r-scharr-sum-nl-single.csv",hough_rd_scharr,delimiter=',')

np.savetxt("hough-circle-accum-sobel-nl-single.csv",hough_accumd_sobel,delimiter=',')
np.savetxt("hough-circle-centre-x-sobel-sum-nl-single.csv",hough_xd_sobel,delimiter=',')
np.savetxt("hough-circle-centre-y-sobel-sum-nl-single.csv",hough_yd_sobel,delimiter=',')
np.savetxt("hough-circle-centre-r-sobel-sum-nl-single.csv",hough_rd_sobel,delimiter=',')
# multi denoise
np.savetxt("hough-circle-accum-scharr-nl-multi.csv",hough_accumd_scharr,delimiter=',')
np.savetxt("hough-circle-centre-x-scharr-sum-nl-multi.csv",hough_xd_scharr,delimiter=',')
np.savetxt("hough-circle-centre-y-scharr-sum-nl-mutli.csv",hough_yd_scharr,delimiter=',')
np.savetxt("hough-circle-centre-r-scharr-sum-nl-multi.csv",hough_rd_scharr,delimiter=',')

np.savetxt("hough-circle-accum-sobel-nl-multi.csv",hough_accumd_sobel_multi,delimiter=',')
np.savetxt("hough-circle-centre-x-sobel-sum-nl-multi.csv",hough_xd_sobel_multi,delimiter=',')
np.savetxt("hough-circle-centre-y-sobel-sum-nl-multi.csv",hough_yd_sobel_multi,delimiter=',')
np.savetxt("hough-circle-centre-r-sobel-sum-nl-multi.csv",hough_rd_sobel_multi,delimiter=',')

## Gaussian blur
print("Trying Gaussian blur on results")
# make directories for results
os.makedirs("ScharrSumDenoise\GaussianBlur",exist_ok=True)
os.makedirs("SobelSumDenoise\GaussianBlur",exist_ok=True)

print("Scharr...")
# current imgs in list are scharr so I'll re use them
for ff,im in enumerate(img_list):
   blur = cv2.GaussianBlur(im,(3,3),0)
   cv2.imwrite(r"ScharrSumDenoise\GaussianBlur\gaussian-blur-scharr-f{}.png".format(ff),blur)
   res = hough_circle(scharr_res_d,radii_range)
   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
   pest_scharr_single[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
   hough_accumd_scharr[ff],hough_xd_scharr[ff],hough_yd_scharr[ff],hough_rd_sobel[ff] = collectHoughStats(accums,cx,cy,radii)

print("Sobel...")
for ff in range(I.shape[2]):
   ## sobel
   sobel_res = np.abs(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,1,0,ksize=5))+np.abs(cv2.Sobel(I_norm_cv[:,:,ff],cv2.CV_8U,0,1,ksize=5))
   # apply denoising
   sobel_res_d = cv2.GaussianBlur(sobel_res,(3,3),0)
   cv2.imwrite(r'SobelSumDenoise\GaussianBlur\gaussian-blur-sobel-f{}.png'.format(ff),sobel_res_d)
   res = hough_circle(sobel_res_d,radii_range)
   accums,cx,cy,radii= hough_circle_peaks(res,radii_range,num_peaks=1)
   pest_sobel_single[ff] = powerEstHoughCircle([cx[accums.argmax()],cy[accums.argmax()],radii[accums.argmax()]],I[:,:,ff],pixel_pitch)
   hough_accumd_sobel[ff],hough_xd_scharr[ff],hough_yd_scharr[ff],hough_rd_sobel[ff] = collectHoughStats(accums,cx,cy,radii)

print("Plotting results")
ax.clear()
ax.plot(pest_scharr_single)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using Gaussian Blurred Results of \n Scharr Edge Detection, Sum')
f.savefig('pest-scharr-sum-gaussian-blur.png')

ax.clear()
ax.plot(pest_sobel_single)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using Gaussian Blurred Results of \n Sobel Edge Detection, Sum')
f.savefig('pest-sobel-sum-gaussian-blur.png')

# plotting hough data
ax.clear()
ax.plot(hough_accumd_scharr)
ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
f.suptitle('Highest Hough Circle Accumulator Score after applying Gaussian Blur\n on the Results of Scharr Edge Detection')
f.savefig('scharr-highest-gaussian-blur-hough-accum.png'.format(file_name))

ax.clear()
ax.plot(hough_xd_scharr)
ax.set(xlabel='Frame Index',ylabel='X Coordinate (Pixels)')
f.suptitle('X Coordinate of Highest Scoring Hough Circle after applying Gaussian Blur\n on the Results of Scharr Edge Detection')
f.savefig('scharr-gaussian-blur-hough-best-circle-centre-x.png'.format(file_name))

ax.clear()
ax.plot(hough_yd_scharr)
ax.set(xlabel='Frame Index',ylabel='Y Coordinate (Pixels)')
f.suptitle('Y Coordinate of Highest Scoring Hough Circle after applying Gaussian Blur\n on the Results of Scharr Edge Detection')
f.savefig('scharr-gaussian-blur-hough-best-circle-centre-y.png'.format(file_name))

ax.clear()
ax.plot(hough_rd_scharr)
ax.set(xlabel='Frame Index',ylabel='Circle Radius (Pixels)')
f.suptitle('Radius of the Highest Scoring Hough Circle after applying Gaussian Blur\n on the Results of Scharr Edge Detection')
f.savefig('scharr-gaussian-blur-hough-best-circle-radius.png')

print("Saving data")
np.savetxt("pest-sobel-sum-gaussian-blur.csv",pest_sobel_single,delimiter=',')
np.savetxt("pest-scharr-sum-gaussian-blur.csv",pest_scharr_single,delimiter=',')
print("Saving Hough Circle stats to file")
# raw
np.savetxt("hough-circle-accum-scharr-gaussian-blur.csv",hough_accumd_scharr,delimiter=',')
np.savetxt("hough-circle-centre-x-scharr-sum-gaussian-blur.csv",hough_xd_scharr,delimiter=',')
np.savetxt("hough-circle-centre-y-scharr-sum-gaussian-blur.csv",hough_yd_scharr,delimiter=',')
np.savetxt("hough-circle-centre-r-scharr-sum-gaussian-blur.csv",hough_rd_scharr,delimiter=',')

np.savetxt("hough-circle-accum-sobel-gaussian-blur.csv",hough_accumd_sobel,delimiter=',')
np.savetxt("hough-circle-centre-x-sobel-sum-gaussian-blur.csv",hough_xd_sobel,delimiter=',')
np.savetxt("hough-circle-centre-y-sobel-sum-gaussian-blur.csv",hough_yd_sobel,delimiter=',')
np.savetxt("hough-circle-centre-r-sobel-sum-gaussian-blur.csv",hough_rd_sobel,delimiter=',')

## try different filtering strengths
os.makedirs("FilterStrengths\Sobel",exist_ok=True)
os.makedirs("FilterStrengths\Scharr",exist_ok=True)
# target frame to use
target_f=1048
# range of strengths to try
h_range = np.arange(10,200,10)
print("Trying different filtering strengths")
for hh in h_range:
    ## sobel
    sobel_res = np.abs(cv2.Sobel(I_norm_cv[:,:,target_f],cv2.CV_8U,1,0,ksize=5))+np.abs(cv2.Sobel(I_norm_cv[:,:,target_f],cv2.CV_8U,0,1,ksize=5))
    # apply denoising
    sobel_res_d = cv2.fastNlMeansDenoising(sobel_res,dst=None,templateWindowSize=10,searchWindowSize=21,h=hh)
    cv2.imwrite("FilterStrengths\Sobel\sobel-denoise-single-h{}-f{}.png".format(int(hh),target_f),sobel_res_d)

    scharr_res = np.abs(cv2.Scharr(I_norm_cv[:,:,target_f],cv2.CV_8U,1,0))+np.abs(cv2.Scharr(I_norm_cv[:,:,target_f],cv2.CV_8U,0,1))
    scharr_res_d = cv2.fastNlMeansDenoising(scharr_res,dst=None,templateWindowSize=10,searchWindowSize=21,h=hh)
    cv2.imwrite("FilterStrengths\Scharr\scharr-denoise-single-h{}-f{}.png".format(int(hh),target_f),scharr_res_d)
print("Finished")
