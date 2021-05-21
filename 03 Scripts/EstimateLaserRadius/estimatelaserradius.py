from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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

def rc_from_cond(d,i):
    ''' Convert condensed matrix indicies to equivalent square form r,c indicides

        d : one side of the dxd square form matrix
        i : indicies to convert

        Returns r,c indicies

        Source: https://stackoverflow.com/a/14839010
    '''
    b = 1-2*d 
    x = np.floor((-b - (b**2 - 8*i)**0.5)/2)
    y = i + x*(b + x + 2)/2 + 1
    return (x,y)

def estR0(M,PP):
    ''' Estimate laser radius from given matrix using 1/e2 method

        M : Matrix to use to calculate matrix
        PP : Pixel Pitch (m)

        Returns the estimated laser radius in terms of metres
    '''
    from scipy.spatial.distance import pdist
    # find the values that are above 1/e2 * max
    x,y = np.where(M>=(1.0/np.exp2(1))*M.max())
    # put the values into a list
    e2_list = list(zip(x,y))
    # tend to get a cluster of values and several outliers
    # need to filter the outliers to make it easier to get the boundary of the cluster
    # searching for locations next to each other
    # calculate the distance between each point and every other point
    dist_cond = pdist(e2_list)
    # if locations are next to eachother the euclidean distance is 1 or sqrt 2
    cond_idx = np.where(np.logical_or(dist_cond==1.0,dist_cond==(2**0.5)))[0]
    # if cluster indicies were found
    # else return 0
    if cond_idx.shape[0]>0:
        # convert the condensed matrix to squareform row,col
        sq_r,sq_c = rc_from_cond(x.shape[0],cond_idx)
        # each row of squareform shows the distance from location r to every other position given
        # e.g. [0,1] would the the distance between point 0 and 1 given in the array of locations provided
        # we need the unique locations. the indicies are concatentated together and the unique values found
        # the result is the indicies of the locations in e2_list that are next to eachother
        clust_loc = [e2_list[p] for p in np.unique(np.concatenate([sq_r.astype('uint16'),sq_c.astype('uint16')]))]
        # need to separate the locations back into rows and columns so they can be used for indexing
        clust_x = [x[0] for x in clust_loc]
        clust_y = [x[1] for x in clust_loc]
        # create empty matrix
        ii = np.zeros(M.shape,dtype=M.dtype)
        # copy cluster values across
        ii[clust_x,clust_y] = M[clust_x,clust_y]
        ## search for boundary of cluster
        ct = cv2.findContours(ii.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
        # if contours were found
        # else return 0
        if len(ct)>0:
            # sort by area in reverse order so the first element is the largest contour
            ct.sort(key=cv2.contourArea,reverse=True)
            # get area of largest contour interms of m2
            ct_area = cv2.contourArea(ct[0])*(PP**2.0)
            # take the area as an area of a circle and work back to radius
            # A = pi*r^2 => (A/pi)**0.5 = r
            return (ct_area/np.pi)**0.5
        else:
            return 0.0
    else:
        return 0.0

def estR0_save(M,PP):
    ''' Estimate laser radius from given matrix using 1/e2 method and save the figures of each stages

        M : Matrix to use to calculate matrix
        PP : Pixel Pitch (m)

        Returns the estimated laser radius in terms of metres
    '''
    from scipy.spatial.distance import pdist
    # find the values that are above 1/e2 * max
    x,y = np.where(M>=(1.0/np.exp2(1))*M.max())
    # create empty matrix
    ii = np.zeros(M.shape,dtype=M.dtype)
    # update values
    ii[x,y] = M[x,y]
    ## create plot
    f,ax = plt.subplots(1,2)
    # original data
    ax[0].contourf(M)
    # values above 1/e2 * peak
    ax[1].contourf(ii)
    # update titles
    ax[0].set_title('Full Dataset')
    ax[1].set_title('Values >= to 1/e2 times peak')
    f.savefig('values-gte-1-e2-peak.png')
    
    # put the values into a list
    e2_list = list(zip(x,y))
    # tend to get a cluster of values and several outliers
    # need to filter the outliers to make it easier to get the boundary of the cluster
    # searching for locations next to each other
    # calculate the distance between each point and every other point
    dist_cond = pdist(e2_list)
    # if locations are next to eachother the euclidean distance is 1 or sqrt 2
    cond_idx = np.where(np.logical_or(dist_cond==1.0,dist_cond==(2**0.5)))[0]
    # if cluster indicies were found
    # else return 0
    if cond_idx.shape[0]>0:
        # draw 1/e2 values
        ax[0].contourf(ii)
        ax[0].set_title('Values >= to 1/e2 times peak')
        # convert the condensed matrix to squareform row,col
        sq_r,sq_c = rc_from_cond(x.shape[0],cond_idx)
        # each row of squareform shows the distance from location r to every other position given
        # e.g. [0,1] would the the distance between point 0 and 1 given in the array of locations provided
        # we need the unique locations. the indicies are concatentated together and the unique values found
        # the result is the indicies of the locations in e2_list that are next to eachother
        clust_loc = [e2_list[p] for p in np.unique(np.concatenate([sq_r.astype('uint16'),sq_c.astype('uint16')]))]
        # need to separate the locations back into rows and columns so they can be used for indexing
        clust_x = [x[0] for x in clust_loc]
        clust_y = [x[1] for x in clust_loc]
        # create empty matrix
        ii = np.zeros(M.shape,dtype=M.dtype)
        # copy cluster values across
        ii[clust_x,clust_y] = M[clust_x,clust_y]
        # draw cluster values on second matrix
        ax[1].contourf(ii)
        # update titles
        f.suptitle('Impact of Cluster Filtering')
        ax[1].set_title('Biggest Cluster of Values')
        f.savefig('values-gte-1-e2-peak-and-biggest-cluster.png')
        ## search for boundary of cluster
        ct = cv2.findContours(ii.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
        # if contours were found
        # else return 0
        if len(ct)>0:
            # sort by area in reverse order so the first element is the largest contour
            ct.sort(key=cv2.contourArea,reverse=True)
            # get area of largest contour interms of m2
            ct_area = cv2.contourArea(ct[0])*(PP**2.0)
            # take the area as an area of a circle and work back to radius
            # A = pi*r^2 => (A/pi)**0.5 = r
            return (ct_area/np.pi)**0.5
        else:
            return 0.0
    else:
        return 0.0

# path to footage
path = "C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"
print("Reading in video data")
frames_set_array = readH264(path)
rows,cols,depth = frames_set_array.shape

target_f = depth # 1049
pixel_pitch=20.0e-6

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

## Perform the estimation process but save each stage
estR0_save(I[:,:,1048],pixel_pitch)

# using target frame 1048
print("I:  {0}".format(estR0(I[:,:,1048],pixel_pitch)))
print("T:  {0}".format(estR0(T[:,:,1048],pixel_pitch)))
print("Qr: {0}".format(estR0(frames_set_array[:,:,1048],pixel_pitch)))

# calculate radius estimate using different matricies
print("Calcuating r0 timelapse using power density matrix")
r0_I = np.array([estR0(I[:,:,f],pixel_pitch) for f in range(I.shape[2])])
print("Plotting result")
curr_cwd = os.getcwd()
f,ax = plt.subplots()
ax.set(xlabel='Frame Index',ylabel='Estimated Laser Radius (m)',title='Estimated Laser Radius')
ax.plot(r0_I)
f.savefig(curr_cwd + '\\r0-I.png')

## find and plot the average value
# find the peak values
r0_peaks_idx = find_peaks(r0_I)[0]
# filter the peak values to just get the more active ones
# filters out smaller peaks occuring near 0
r0_peaks_idx = r0_peaks_idx[r0_I[r0_peaks_idx]>0.05*np.max(r0_I[r0_peaks_idx])]
# get the actual peak values
r0_peaks = r0_I[r0_peaks_idx]
## the peaks have two major outliers so we just want the majority of the values
# calculate standard deviation of the peak values
r0_std = np.std(r0_peaks)
# calculate the mean of the peaks
r0_mean = np.mean(r0_peaks)
# just get the peaks that are within 3 stds of the mean
r0_no_out = np.where(np.logical_and(r0_peaks>=(r0_mean-3.0*r0_std),r0_peaks<=(r0_mean+3.0*r0_std)))
r0_filt_mean = np.mean(r0_peaks[r0_no_out])

# plot the results
f,ax = plt.subplots()
ax.set(xlabel='Frame Index',ylabel='Estimated Laser Radius (m)',title='Estimated Laser Radius using the Power Density Matrix')
ax.plot(r0_I,'b-', # original data
        r0_peaks_idx,r0_peaks,'yx', # all peaks after first filter
        r0_peaks_idx[r0_no_out],r0_peaks[r0_no_out],'rx' # peaks within 3 std
        ,np.arange(0,depth),np.full(I.shape[2],r0_filt_mean),'g-') # line showing mean of result
f.legend(['Est. r0','Peak r0','Peak r0 (3 std.)','Avg. r0={:.4f}mm'.format(r0_filt_mean*10**3.0)])
f.savefig(curr_cwd + '\\r0-peaks-mean.png')



