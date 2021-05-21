from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.signal import find_peaks
from scipy.fftpack import fftn,ifftn,fftshift
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage import color
import os
import cv2

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
   
# path to footage
path = "D:/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"
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

print("Normalizing power density matricies")
# norm matricies
I_norm = I/I.max(axis=(0,1))
print("Creating folders")
# create folders for results
os.makedirs('Norm-circles',exist_ok=True)

print("Searching for circles in 8-bit image")
# skimage hough transform requires a list of circles to search for
# using the same range as the opencv version
radii_range = np.arange(0,4*r0_p,0.1)

# using test frame 1048
# the results are a 3d arrray of values listing the circles found for
# each target radii
res = hough_circle(I[:,:,1048].astype('uint8'),radii_range)

# select the 5 most prominent circles
# accums is a measure of how the circles are ranked by the Hough Circle algorithm
# hough_circle_peaks searches for the most highly ranked
accums, cx,cy,radii = hough_circle_peaks(res,radii_range,total_num_peaks=5)
# create copy of mask to draw results on
mask=color.gray2rgb(I[:,:,1048].astype('uint8'))
# iterate through circles
for x,y,r in zip(cx,cy,radii):
    # get the y,x coordinates of the circle perimeter
    circy,circx = circle_perimeter(y,x,int(r),shape=I.shape[:2])
    # "draw" the borders on the image 
    mask[circy,circx]=(220,20,20)
    
f,ax = plt.subplots()
# show image
ax.imshow(mask,cmap=plt.cm.gray)
f.savefig('skimage-hough-circles-eight-bit-cast-I.png')

print("Searching for circles in normalized image")
## applying it to the normalized matrix
res = hough_circle(I_norm[:,:,1048],radii_range)
accums, cx,cy,radii = hough_circle_peaks(res,radii_range,total_num_peaks=5)
# create copy of mask to draw results on
mask=color.gray2rgb(I_norm[:,:,1048])
# iterate through circles
for x,y,r in zip(cx,cy,radii):
    # get the y,x coordinates of the circle perimeter
    circy,circx = circle_perimeter(y,x,int(r),shape=I.shape[:2])
    # "draw" the borders on the image 
    mask[circy,circx]=(220.0/255,20.0/255,20.0/255)
    
ax.clear()
# show image
ax.imshow(mask,cmap=plt.cm.gray)
f.savefig('skimage-hough-circles-norm-I.png')

## summing the areas of circles to find the best estimate
# search for hough circles
crange = list(range(1,10))
pest = np.zeros(len(crange))
res = hough_circle(I_norm[:,:,1048],radii_range)
for clim in crange:
   print(clim)
   # find a certain number of the best circles
   accums,cx,cy,radii = hough_circle_peaks(res,radii_range,total_num_peaks=clim)
   ## draw them onto a mask
   mask = np.zeros((rows,cols),np.uint8)
   for x,y,r in zip(cx,cy,radii):
      cv2.circle(mask,(int(x),int(y)),int(r),(255),-1)
   # search for where the mask is set
   xx,yy = np.where(mask==255)
   # write drawn mask to file
   cv2.imwrite('sum-hough-circles-clim-{0}.png'.format(clim),mask)
   # search for contour of the area
   cc = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
   # calculate power estimate of the largest contour
   pest[clim-1]=np.sum(I[xx,yy,1048])*cv2.contourArea(cc[0])*(pixel_pitch**2.0)

f,ax = plt.subplots()
ax.plot(crange,pest)
ax.set(xlabel='Number of Circles Summed',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using the Sum Area of Circles Found')
f.savefig('pest-sum-circles-houghcircles.png')

## Analyse skimage parameters
radii_range = np.arange(0,4*r0_p,1)
# number of circles to draw
draw_ci = 10
## power estimates
# largest circle
pest_large = np.zeros(I.shape[2])
# best circle
pest_best = np.zeros(I.shape[2])
# circle with the highest accumulated score
pest_accum = np.zeros(I.shape[2])
# best power estimate after the circles have been filtered
pest_best_filt = np.zeros(I.shape[2])
# power estimate using the largest circle after it has been filtered
pest_large_filt = np.zeros(I.shape[2])
## radii
# largest radius
r_large = np.zeros(I.shape[2])
# largest radius after the circles have been filtered
r_large_filt = np.zeros(I.shape[2])
# radius of the best circle
r_best = np.zeros(I.shape[2])
# radius of the circle with the highest accumulated score
r_accum = np.zeros(I.shape[2])
## radii ratio
# ratio between the largest circle radius and the laser radius
rratio_large = np.zeros(I.shape[2])
# ration between the best circle radius and the laser radius
rratio_best = np.zeros(I.shape[2])
# ratio between the circle with the highest score and the laser radius
rratio_accum = np.zeros(I.shape[2])
# ratio between the largest circle after filtering and the laser radius
rratio_large_filt = np.zeros(I.shape[2])
## accumulator score
# score of the best circle
accum_best = np.zeros(I.shape[2])
# highest accumulator score
accum_high = np.zeros(I.shape[2])
## number of promiment circles found
num_circles = np.zeros(I.shape[2],np.uint32)

print("Starting large analysis")
# iterate through each normalized frame
for ff in range(I.shape[2]):
   print("Frame ",ff)
   # perform skimage hough circle
   res = hough_circle(I_norm[:,:,ff],radii_range)
   # get the most prominent circles, removes low accumulator score circles
   accums,cx,cy,radii = hough_circle_peaks(res,radii_range)
   ## collect metrics
   r_large[ff] = radii.max()
   r_accum[ff] = radii[accums.argmax()]
   rratio_large[ff] = r_large.max()/r0_p
   rratio_accum[ff] = r_accum[ff]/r0_p
   accum_high[ff] = accums.max()
   num_circles[ff] = cx.shape[0]
   # the power estimate function is based off the results for OpenCV's Hough Circle function
   # the results from the skimage Hough Circle function have to be repackaged
   circles = np.array([[[x,y,r] for x,y,r in zip(cx,cy,radii)]],dtype='float32')
   # for each circle analyse its power and store the results in a numpy array
   pest = np.array([powerEstHoughCircle(c,I[:,:,ff],pixel_pitch) for c in circles[0]])
   pest_large[ff] = pest[radii.argmax()]
   # idx of circle that gives the closest estimate to 500.0W
   best_ci = np.abs(500.0-pest).argmin()
   pest_best[ff] = pest[best_ci]
   pest_accum[ff] = pest[accums.argmax()]
   r_best[ff] = radii[best_ci]
   rratio_best[ff] = r_best[ff]/r0_p
   accum_best[ff] = accums[best_ci]
   # filter out of bound circles
   circles_filt = remCirclesOutBounds(circles,(rows,cols))
   pest = np.array([powerEstHoughCircle(c,I[:,:,ff],pixel_pitch) for c in circles_filt[0]])
   # gather metrics for filtered circles
   r_large_filt[ff] = circles_filt[0,:,2].max()
   rratio_large_filt[ff] = r_large_filt[ff]/r0_p
   pest_best_filt[ff] = pest[np.abs(500.0-pest).argmin()]
   pest_large_filt[ff] = powerEstHoughCircle(circles_filt[0,circles_filt[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)
   
   # draw ten best circles
   mask = drawHoughCircles(circles,(rows,cols),num_c=draw_ci)
   # write result to file
   cv2.imwrite('Norm-circles/norm-circles-f{0}.png'.format(ff),mask)

# plot the results
# it's a separate function so that if the user Ctrl+C force stops the program, they can still plot the results
# by calling the function.
def plotResults():
   # create plot axes
   f,ax = plt.subplots()
   # create power estimates
   ax.plot(pest_large[:ff])
   ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
   f.suptitle('Power Estimate Using the Largest Circle Found using Skimage Hough Circle')
   f.savefig('skimage-pest-large-circle.png')

   ax.clear()
   ax.plot(pest_best[:ff])
   ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
   f.suptitle('Power Estimate Using the Best Circle Found using Skimage Hough Circle')
   f.savefig('skimage-pest-best-circle.png')

   ax.clear()
   ax.plot(pest_accum[:ff])
   ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
   f.suptitle('Power Estimate Using the Circle with the Highest Accumulator Score')
   f.savefig('skimage-pest-accum-circle.png')

   ax.clear()
   ax.plot(pest_best_filt[:ff])
   ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
   f.suptitle('Power Estimate Using the Best Circle After Filtering Circles \n that are Out of Bounds')
   f.savefig('skimage-pest-best-filt-circle.png')

   ax.clear()
   ax.plot(pest_large_filt[:ff])
   ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
   f.suptitle('Power Estimate Using the Largest Circle After Filtering Circles \n that are Out of Bounds')
   f.savefig('skimage-pest-large-filt-circle.png')

   # create radius
   ax.clear()
   ax.plot(r_large[:ff])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   f.suptitle('Radius of the Largest Circle Found using Skimage Hough Circle')
   f.savefig('skimage-radius-large-circle.png')

   ax.clear()
   ax.plot(r_large_filt[:ff])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   f.suptitle('Radius of the Largest Circle After Filtering Circles \n that are Out of Bounds')
   f.savefig('skimage-radius-large-filt-circle.png')

   ax.clear()
   ax.plot(r_best[:ff])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   f.suptitle('Radius of the Best Circle Found using Skimage Hough Circle')
   f.savefig('skimage-radius-best-circle.png')

   ax.clear()
   ax.plot(r_accum[:ff])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   f.suptitle('Radius of the Circle with the Highest Accumulator Score')
   f.savefig('skimage-radius-accum-circle.png')

   # radius ratio
   ax.clear()
   ax.plot(rratio_large[:ff])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   f.suptitle('Ratio of Radius of the Largest Circle Found using Skimage Hough Circle to Laser Radius')
   f.savefig('skimage-rratio-large-circle.png')

   ax.clear()
   ax.plot(rratio_large_filt[:ff])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   f.suptitle('Ratio of Radius of the Largest Circle to Laser Radius \n After Filtering Circles that are Out of Bounds to Laser Radius')
   f.savefig('skimage-rratio-large-filt-circle.png')

   ax.clear()
   ax.plot(rratio_best[:ff])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   f.suptitle('Ratio of Radius of the Best Circle Found using Skimage Hough Circle to Laser Radius')
   f.savefig('skimage-rratio-best-circle.png')

   ax.clear()
   ax.plot(rratio_accum[:ff])
   ax.set(xlabel='Frame Index',ylabel='Laser Radius (Pixels)')
   f.suptitle('Ratio of Radius of the Circle with the Highest Accumulator Score to Laser Radius')
   f.savefig('skimage-rratio-accum-circle.png')

   # accumulator score
   ax.clear()
   ax.plot(accum_best[:ff])
   ax.set(xlabel='Frame Index',ylabel='LAccumulator Score')
   f.suptitle('Hough Circle Accumulator Score of the Circle \n with the Closest Power Estimate')
   f.savefig('skimage-accum-best-circle.png')

   ax.clear()
   ax.plot(accum_high[:ff])
   ax.set(xlabel='Frame Index',ylabel='Accumulator Score')
   f.suptitle('Highest Hough Circle Accumulator Score')
   f.savefig('skimage-accum-high-circle.png')

   # number of circles
   ax.clear()
   ax.plot(num_circles[:ff])
   ax.set(xlabel='Frame Index',ylabel='Number of Circles')
   f.suptitle('Number of Prominent Circles Found Using Skimage Hough Circle')
   f.savefig('skimage-num-dominant-circles.png')

print("Starting estimation of power")
# wide range
radii_range = np.arange(30,70)
# narrow range
radii_range_narr = np.arange(60,70)
# power estimate using different methods
pest = np.zeros(I.shape[2])
pest_accums = np.zeros(I.shape[2])
pest_narrow = np.zeros(I.shape[2])

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
   
# list comprehension power estimate
pest = [powerEstBestCircle(I[:,:,ff],radii_range,pixel_pitch) for ff in range(I.shape[2])]
pest_accums = [powerEstAccumCircle(I[:,:,ff],radii_range,pixel_pitch,min_accum=0.6) for ff in range(I.shape[2])]
pest_narrow = [powerEstAccumCircle(I[:,:,ff],radii_range_narr,pixel_pitch) for ff in range(I.shape[2])]

f,ax = plt.subplots()
ax.plot(pest)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using Highest Scoring Hough Circle Function')
f.savefig('skimage-hough-circle-pest-accums-best.png')

ax.clear()
ax.plot(pest_accums)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using Hough Circle with Highest Score Above Min, t=0.6')
f.savefig('skimage-hough-circle-pest-accums-min.png')

ax.clear()
ax.plot(pest_narrow)
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.suptitle('Power Estimate Using Hough Circle with Highest Score Above Min \n and Narrow Radii Range, t=0.6')
f.savefig('skimage-hough-circle-pest-accums-min-narrow-radii.png')
