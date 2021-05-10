from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from TSLoggerPy.TSLoggerPy import TSLoggerPy as logger
import cv2
import os
import cmapy
import time

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

        Returns numpy array surface temperature in Kelvin using Stefan-Boltzman theory in W/m2

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

def estPArea_Canny(I,PP,lt=100,r=3):
    """ Estimating Laser Power by Area using Contours to estimate Area

        I : Power density image (W/pixel)
        PP : Pixel pitch (m)

        The power density image is filtered using Canny edge detection.
        The result is fed into findContours

        Returns estimated power or 0.0 if it can't find any contours
    """
    import cv2
    from numpy import zeros_like,where
    from numpy import sum as nsum
    # convert data to image
    # eq. to grayscale
    I_img = np.uint8(I)
    #print(I_img.dtype)
    # threshod using otsu, returns threshold and thresholded img
    #lt = 100
    #r = 3
    # apply blur to image
    I_blur = cv2.blur(I_img,(3,3))
    # apply canny edge detection to create mask
    edges = cv2.Canny(I_blur,lt,lt*r,3)
    # find contours, returne contours and hierarchy
    ct=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
    if len(ct)>0:
        # sort contours by area in ascending order
        ct = sorted(ct,key=lambda x : cv2.contourArea(x))
        ct_idx = 0
        # if the largest contour is likely the outer edge of the image
        if cv2.contourArea(ct[-1]) >= 0.9*(I.shape[0]*I.shape[1]):
            # use next largest
            largest_area = cv2.contourArea(ct[-2])
            ct_idx = -2
        else:
            largest_area = cv2.contourArea(ct[-1])
            ct_idx = -1
        # draw contour of interest on mask filled
        mask = cv2.drawContours(np.zeros(I_img.shape,dtype=I_img.dtype),ct,ct_idx,255,-1)

        # create color version of mask to return
        I_col = cv2.cvtColor(I_img,cv2.COLOR_GRAY2BGR)
        # draw masked area in bright green
        cv2.drawContours(I_col,ct,ct_idx,(0,255,0),-1)
        #print(largest_area,img_area)
        # sum the energy densities and multiply by area to get total energy
        # only sums the places where mask is 255
        return edges,I_col,nsum(I[where(mask==255)])*largest_area*(PP**2.0)
    else:
        return edges,zeros_like(I_img),0.0
    
pixel_pitch = 20.0*10.0**-6 # metres

# path to footage
path = "C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"
print("Reading in video data")
frames_set_array = readH264(path)
rows,cols,depth = frames_set_array.shape

target_f = depth # 1049

fc = 65.0
tc = 1/fc

T = np.zeros(frames_set_array.shape,dtype='float64')
I = np.zeros(frames_set_array.shape,dtype='float64')

print("Evaluating up to ", target_f)
for f in range(0,target_f,1):
    #print("f=",f)
    if f==0:
        T[:,:,f] = Qr2Temp(frames_set_array[:,:,f],e_ss,T0)
        I[:,:,f] = Temp2IApprox(T[:,:,f],T0,Kp_spline,Ds_spline,tc)
    else:
        # prev temp has to be T[:,:,f] to get expected behaviour for Tpeak
        # not sure why
        # if T0 is T[:,:,f-1], then Tpeak rises from 500K to 3200K
        T[:,:,f] = Qr2Temp(frames_set_array[:,:,f],e_ss,T[:,:,f])
        I[:,:,f] = Temp2IApprox(T[:,:,f],T[:,:,f-1],Kp_spline,Ds_ispline,tc)
        
    if np.inf in I[:,:,f]:
        np.nan_to_num(I[:,:,f],copy=False)
      
print("Finished on frame ", f)
   
P_actual = 500

pitch_sq = pixel_pitch**2.0
# Using an example active frame
I_img = I[:,:,1048].astype('uint8')
# ratio between lower and upper threshold for Canny
# if the intensity gradient is above the upper threshold, then it's an edge
# if it's below the intensity gradient then it's discarded
#r_range = 1.0+(np.arange(1,10000,100)/1000)
r_range = np.arange(1.0,4.0,0.01)
# lower threshold range
lt_range = np.arange(0,200,1)

# from visual inspection a narrower range  of parameters was identified
# finds where in the current range to start from 
lt_narr = 200
r_narr = 5.0
# try and search for an exact match
rn_idx = np.where(r_range==r_narr)[0]
# if an exact match cannot be found
if rn_idx.shape[0]==0:
   # sort the r_range by distance from target value
   rn_idx = np.argsort(np.abs(r_range-r_narr))[0]
else:
   rn_idx

ltn_idx = np.where(lt_range==lt_narr)[0]
if ltn_idx.shape[0]==0:
   ltn_idx = np.argsort(np.abs(lt_range-lt_narr))[0]
else:
   ltn_idx = ltn_idx[0]
   
# create 2D coordinate matricies for the entire range    
lt_m,r_m = np.meshgrid(lt_range,r_range)

# power estimate
# shows impact of lower threshold and ratio adjustements on Canny estimate on power
P_est = np.zeros((r_range.shape[0],lt_range.shape[0]),dtype=I.dtype)
# contour idx that gives the closest power estimate
pbest_ct = np.zeros((r_range.shape[0],lt_range.shape[0]),dtype='uint16')
# matrix to test different methods
pbest_ct_find = np.zeros((r_range.shape[0],lt_range.shape[0]),dtype='uint16')

fig_size = [x*4 for x in plt.rcParams['figure.figsize']]

print("Power estimate matrix: ",P_est.shape)

def findBestPowerCt(ct,I,P_actual,pixel_pitch):
   ''' Find the contour that will give the closest power estimate

       ct : list of contours returned by findContours
       I : power density frame
       P_actual : Target power estimate (W)
       pixel_pitch: pixel pitch (m)

       Returns the index of the best contour
   '''
   # enumerate contours
   P_dist = 0.0
   c_ci = 0.0
   for ci,c in enumerate(ct):
      # calculate area of contour
      ct_area = cv2.contourArea(c)
      # clear mask
      mask = np.zeros_like(I.astype('uint8'))
      # draw largest contours on mask with 255 values
      cv2.drawContours(mask,ct,ci,255,-1)
      # find where mask is set
      x,y = np.where(mask==255)
      ## calculate power using target frame
      # find total power density
      I_sum = np.sum(I[x,y])
      # multiply sum by area converted to metres^2
      P_temp = I_sum* (ct_area*pixel_pitch**2.0)
      # if the power estimate for this contour is closer than the current min
      # store new min distance and index
      if np.abs(P_temp - P_actual) < P_dist:
         P_dist = np.abs(P_temp - P_actual)
         c_ci = ci
   return int(c_ci)

def ct2Power(ct,I,PP):
   ''' Calculate the power for given contour using laser power density frame

       ct : Single contour
       I : Target power density frame
       PP : pixel pitch, used to calculate area

       Returns the estimated power within the contour
   '''
   # draw contour on empty matrix to form mask
   mask = cv2.drawContours(np.zeros(I.shape,dtype='uint8'),[ct],0,255,-1)
   # search for where mask was set
   x,y = np.where(mask==255)
   # calculate power by summing the power density values within the contour
   # and multiplying it by the area in terms of m2
   # W/m2 * m2 -> W
   return np.sum(I[x,y])*cv2.contourArea(ct)*PP

# iterate through combinations of ration and lower threshold for known active frame 1048
print("Processing lower threshold-ratio matrix")
t0 = time.time()
for ri,r in enumerate(r_range):
   for lti,lt in enumerate(lt_range):
##      # perform canny edge detection with 3x3 Sobel operator
##      edges = cv2.Canny(I_img,lt,lt*r,3)
##      # find contours, return contours and hierarchy
##      ct=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
##      # if contours were found
##      if len(ct)>0:
##         # sort in descending order, first element is largest
##         ct.sort(key=cv2.contourArea,reverse=True)
##         # calculate area of largest contour
##         ct_area = cv2.contourArea(ct[0])
##         # draw largest contours on mask with 255 values
##         mask = cv2.drawContours(np.zeros(I_img.shape,dtype=I_img.dtype),ct,0,255,-1)
##         # find where mask is set
##         x,y = np.where(mask==255)
##         # calculate power of target frame
##         # sum the power densities within the contour and multiply be area
##         # W/m2 * m2 -> W
##         P_est[ri,lti] = np.sum(I[x,y,1048])*(ct_area*pitch_sq)
         P_est[ri,lti]= estPArea_Canny(I[:,:,f],pixel_pitch,lt=lt,r=r)
         # clear stored min distance and contour index
         P_dist = 0.0
         c_ci = 0
         # enumerate contours
         for ci,c in enumerate(ct):
            # calculate area of contour
            ct_area = cv2.contourArea(c)
            # draw largest contours on mask with 255 values
            mask = cv2.drawContours(np.zeros(I_img.shape,dtype=I_img.dtype),[c],0,255,-1)
            # find where mask is set
            x,y = np.where(mask==255)
            # calculate power using target frame
            P_temp = np.sum(I[x,y,1048])*(ct_area*pitch_sq) 
            # if the power estimate for this contour is closer than the current min
            # store new min distance and index
            if np.abs(P_temp - P_actual) < P_dist:
               P_dist = np.abs(P_temp - P_actual)
               c_ci = ci
         # update ideal contour given Canny settings matrix
         pbest_ct[ri,lti]=ci

print("Time taken: ",time.time()-t0)
print("Finished")
print("Plotting...")
# setting all inf values to 2000W
P_est[np.isinf(P_est)]=2000.0

X,Y = np.meshgrid(np.arange(0,rows),np.arange(0,cols))
# plot power estimate 
f = plt.figure(figsize=fig_size)
ax = f.add_subplot(121)
div = make_axes_locatable(ax)
cax = div.append_axes('right',size='5%',pad = 0.05)
ct_P = ax.contourf(lt_m,r_m,P_est)
f.colorbar(ct_P,cax=cax,orientation='vertical')
ax.set(xlabel='Canny LT',ylabel='Ratio between LT and UT',title='Impact of Different Canny Parameters on the Power')

# plot power density matrix
ax2 = f.add_subplot(122)
div = make_axes_locatable(ax2)
cax = div.append_axes('right',size='5%',pad = 0.05)
ct_I = ax2.contourf(I[:,:,1048])
f.colorbar(ct_I,cax=cax,orientation='vertical')
ax2.set(xlabel='X',ylabel='Y',title='Laser Power Density')

f.savefig('C:/Users/DB/Desktop/BEAM/Scripts/PowerCannyEval/power-est-canny-params-max-cont-wI-{0}.png'.format(I.dtype))

# plot ideal contour plot
f,ax = plt.subplots()
div = make_axes_locatable(ax)
cax = div.append_axes('right',size='5%',pad = 0.05)
ct_ct = ax.contourf(lt_m,r_m,pbest_ct)
f.colorbar(ct_ct,cax=cax,orientation='vertical')
ax.set(xlabel='Canny LT',ylabel='Ratio between LT and UT',title='Best Contour given Different Canny Parameters')
f.savefig('C:/Users/DB/Desktop/BEAM/Scripts/PowerCannyEval/best-cont-canny-params-{0}.png'.format(I.dtype))

## find the best parameters
# number of parameters to find, i.e. N best parameter set
N = 5
# flatten the difference array and sort
# convert the first N sorted incides back to 2d indicies
best_idx = np.unravel_index(np.argsort(np.abs(500.0-P_est).ravel(),axis=None)[:N],P_est.shape)
# returned indicies are two separate arrays, x,y
print(N," chosen best values")
for x,y in zip(best_idx[0],best_idx[1]):
   print("r:{:.2f}, lt:{:.2f}".format(r_range[x],lt_range[y]))

# get the values
best_r = r_range[best_idx[0]]
best_lt = lt_range[best_idx[1]]

fig,ax = plt.subplots(1,2,constrained_layout=True)
fig.suptitle('Power Estimates using different Canny Parameters')
# power estimate using the parameters
P_est_param = np.zeros(I.shape[2])
# power estimate with default parameters
# calculated outside as it doesn't change
print("Calculating original power estimate")
pest_orig = np.array([estPArea_Canny(I[:,:,ff],pixel_pitch)[2] for ff in range(depth)],dtype=I.dtype)
# replace nans and infs in power estimate
# tends to happen if predicition data type is float16
if (np.inf in pest_orig) or (np.nan in pest_orig):
   # replace infs with 2000
   pest_orig[np.isinf(pest_orig)]=2000.0
   # replace nans with 0.0
   pest_orig[np.isnan(pest_orig)]=0.0

# set labels on axis
ax[0].set(xlabel='Frame Index',ylabel='Estimated Power (W)',title='Original Power Estimate')
ax[1].set(xlabel='Frame Index',ylabel='Estimated Power (W)')
# plot original data estimate
ax[0].plot(pest_orig)
# create line object using original power estimate
# will be updated with power estimate using new Canny parameters
param_line, = ax[1].plot(pest_orig)
I_dtype_str = str(I.dtype)
print("Calculating power estimate for the ",N," paramter pairs")
for r,lt in zip(best_r,best_lt):
   for f in range(depth):
##      I_img = I[:,:,f].astype('uint8')
##      # perform canny edge detection with 3x3 Sobel operator
##      edges = cv2.Canny(I_img,lt,lt*r,3)
##      # find contours, return contours and hierarchy
##      ct=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
##      # if contours were found
##      if len(ct)>0:
##         # sort in descending order, first element is largest
##         ct.sort(key=cv2.contourArea,reverse=True)
##         # calculate area of largest contour
##         ct_area = cv2.contourArea(ct[0])
##         # draw largest contours on mask with 255 values
##         mask = cv2.drawContours(np.zeros(I_img.shape,dtype=I_img.dtype),ct,0,255,-1)
##         # find where mask is set
##         x,y = np.where(mask==255)
##         # calculate power of target frame
##         # sum the power densities within the contour and multiply be area
##         # W/m2 * m2 -> W
##         P_est_param[f] = np.sum(I[x,y,f])*(ct_area*pitch_sq)
      P_est_param[f] = estPArea_Canny(I[:,:,f],pixel_pitch,lt,r)[2]
   ## plot parameters and adjust axis title
   # update line data
   param_line.set_ydata(P_est_param)
   # update title
   ax[1].set_title('Power Estimate, r={:.2f},lt={:.2f}'.format(r,lt))
   # save figure with path indicating the parameters used and the data type (16 or 64 float)
   fig.savefig('power-est-compare-r{0}-{1}-lt{2}-{3}-t{4}.png'.format(*'{:.2f}'.format(r).split('.'),*'{:.2f}'.format(lt).split('.'),I_dtype_str))

#plt.show()
   
##
#### get est power for all power desities
### create axes using last Power v Canny matrix
##f,ax = plt.subplots(figsize=fig_size)
##cax_P = make_axes_locatable(ax).append_axes('right',size='5%',pad = 0.05)
##ct_P = ax.contourf(r_m,lt_m,P_est)
##ax.set(xlabel='Canny LT',ylabel='Ratio between LT and UT')
##tx = ax.set_title('Impact of Different Canny Parameters on the Power Estimate')
##
#### update ideal contour plot
### create axes using last ideal contour plot
##f2,ax2 = plt.subplots(figsize=fig_size)
##cax_ct = make_axes_locatable(ax2).append_axes('right',size='5%',pad = 0.05)
##ct_ct = ax2.contourf(r_m,lt_m,pbest_ct)
##ax2.set(xlabel='Canny LT',ylabel='Ratio between LT and UT')
##tx2 = ax2.set_title('Impact of Different Canny Parameters on the Ideal Contour')


##f,ax = plt.subplots(1,1)
##P_ct = ax.contourf(P_est)
##cax_P = make_axes_locatable(ax).append_axes('right',size='5%',pad='2%')
##f2,ax2 = plt.subplots()
##ct_ct = ax.contourf(pbest_ct)
##cax_ct = make_axes_locatable(ax).append_axes('right',size='5%',pad='2%')
##
##print("Creatng power estimate timelapse")
##import time
##t0 = time.time()
##for fi in range(depth):
##   print("Starting on frame ", fi)
##   I_img = I[:,:,fi].astype('uint8')
##   for ri,r in enumerate(r_range[rn_idx:]):
##      for lti,lt in enumerate(lt_range[ltn_idx:]):
##         #print("Finding contours")
##         # find contours in the image once filtered by Canny edge detection, return contours and hierarchy
##         ct=cv2.findContours(cv2.Canny(I_img,lt,lt*r,3),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
##         # if contours were found
##         if len(ct)>0:
##            # the power 
##            # as the areas were
##            # use largest 
##            # sort in descending order, first element is largest
##            #ct = sorted(ct,key=cv2.contourArea,reverse=True)
##            ct.sort(key=cv2.contourArea,reverse=True)
##            # calculate area of each contour and save as list
##            ct_area = [cv2.contourArea(c) for c in ct]
##            ## find power estimate for largest contour
##            # clear mask
##            mask = np.zeros_like(I_img)
##            # fill area defined by contour with value 255
##            cv2.drawContours(mask,ct,0,255,-1)
##            # find where mask is set
##            x,y = np.where(mask==255)
##            # calculate power using target frame
##            I_sum = np.sum(I[x,y,fi])
##            I_sum *= ct_area[0]
##            # estimate power
##            P_est[ri,lti] = I_sum * pitch_sq
##            # find the best contour for the frame that will give the closest power estimate
##            pbest_ct[ri,lti]=findBestPowerContour(ct,I[:,:,fi],P_actual,pixel_pitch)
##
##   # replace inf power estimates with known max of 2k
##   P_est[np.isinf(P_est)]=2000.0
##
##   ## save results as images rather than full plots. MUCH FASTER
##   # apply custom colormap
##   P_est_col = cv2.applyColorMap(P_est.astype('uint8'),cmapy.cmap('viridis'))
##   cv2.imwrite('C:/Users/DB/Desktop/BEAM/Scripts/PowerCannyEval/Canny Params Contours/Power Estimate/power-est-canny-params-max-cont-f{0}.png'.format(fi), P_est_col.astype('uint16'))
##
##   pbest_ct_col = cv2.applyColorMap(pbest_ct.astype('uint8'),cmapy.cmap('viridis'))
##   cv2.imwrite('C:/Users/DB/Desktop/BEAM/Scripts/PowerCannyEval/Canny Params Contours/Ideal Contour/Sorted/ideal-cont-canny-params-f{0}.png'.format(fi),pbest_ct_col.astype('uint16'))
##
##   # create new power estimate contour
##   # scale power estimate to known limits
##   P_ct = ax.contourf(P_est,vmin=0.0,vmax=P_actual)
##   plt.colorbar(P_ct,cax=cax_P)
##   # update title
##   ax.set_title('Impact of Different Canny Parameters on the Power Estimate, Frame {0}'.format(fi))
##   # save figure
##   f.savefig('C:/Users/DB/Desktop/BEAM/Scripts/PowerCannyEval/Canny Params Contours/Power Estimate/power-est-canny-params-max-cont.f{0}.png'.format(fi)) 
##
##   # create new contour
##   ct_ct = ax2.contourf(pbest_ct)
##   plt.colorbar(ct_ct,cax=cax_ct)
##   # update title
##   ax2.set_title('Impact of Different Canny Parameters on the Power Estimate, Frame {0}'.format(fi))
##   # save figure
##   f2.savefig('C:/Users/DB/Desktop/BEAM/Scripts/PowerCannyEval/Canny Params Contours/Ideal Contour/Sorted/ideal-cont-canny-params-f{0}.png'.format(fi))
