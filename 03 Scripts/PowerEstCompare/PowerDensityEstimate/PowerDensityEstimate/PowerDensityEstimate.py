from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.offsetbox import AnchoredText
import os
from TSLoggerPy.TSLoggerPy import TSLoggerPy as logger

#log = logger()
#logger.closeLog()
# hange cwd to file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("Starting")

# path to footage
path = "C:/Users/DB/Desktop/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"

def readH264(path,flag='mask'):
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
    # combined frames together into a 3d array
    return np.dstack([np.reshape(f,(rows,cols)) for f in frames_set])

print("Reading in thermal data")
# read in thermal camera data
frames_set_array = readH264(path)

#def readAbsorbParams(folderpath):
#    """ Read absorbtivity parameters written to file into a dictionary

#        folderpath : Path to the folder where the documents are

#        Returns a dictionary of the numpy.polynomial.polynomial.Polynomial objects
#        for the parameters

#        Dictionary keys are:
#            "growth" : growth rate of sigmoid for range of velocities
#            "lower"  : lower asymptote of sigmoid
#            "upper"  : upper asymptote of sigmoid
#            "Q"      : Q-factor of sigmoid
#            "v"      : v-factor of sigmoid
#    """
#    from numpy import genfromtxt
#    import numpy.polynomial.polynomial as poly

#    # create dictionary object
#    params_dict = {}
#    # read in parameters from file as array and convert to list
#    params = genfromtxt(folderpath+'/absorb-poly-growth-params.txt',delimiter=',').tolist()
#    # construct polynomial object, only using 3 values as 4th is nan
#    params_dict["growth"] = poly.Polynomial(params[:3])

#    params= genfromtxt(folderpath+'/absorb-poly-lower-params.txt',delimiter=',').tolist()
#    params_dict["lower"] = poly.Polynomial(params[:3])

#    params= genfromtxt(folderpath+'/absorb-poly-upper-params.txt',delimiter=',').tolist()
#    params_dict["upper"] = poly.Polynomial(params[:3])

#    params = genfromtxt(folderpath+'/absorb-poly-Q-params.txt',delimiter=',').tolist()
#    params_dict["Q"] = poly.Polynomial(params[:3])

#    params= genfromtxt(folderpath+'/absorb-poly-v-params.txt',delimiter=',').tolist()
#    params_dict["v"] = poly.Polynomial(params[:3])
#    # return poly dictionary
#    return params_dict

#print("Reading in absorbtivity parameters")
#absob_dict = readAbsorbParams('C:/Users/DB/Desktop/BEAM/Scripts/AbsorbtivityData/AbsorbtivityData')

# remove artifact on the left hand side
mask_width = 3
frames_set_array[:,:mask_width,:] = 0
# get size of the 
rows,cols,num_frames = frames_set_array.shape

print("Converting data from uint16 to float 16")
# interpret values as float uint16 -> float16
frames_set_array = frames_set_array.astype(dtype=np.float16)

# sampling rate of thermal camera
Fc = 65.0 # hz
# tc used as interaction time
tc = 1.0/Fc # s

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

Kp_data = Kp(K_data,K_temp_range,Tm)
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
        p_range_2.append(p_range_1[0]/(1+Ev(dT)))
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

### plot results as subplots
#param_fsize = (12.0, 8.0)
## thermal conductivity
#print("Plotting parameter data")
#f,((axK_1,axK_2,axK_3),(axK_4,axK_5,axK_6)) = plt.subplots(2,3,tight_layout=True,figsize=param_fsize)
#f.suptitle("Thermal Conductivity Spline Prediction")
## solid
#axK_1.plot(K_temp_range,K_data,'b+',label='data')
#axK_1.plot(T_test_data,K_spline(T_test_data),'g-',label='S-Spline')
#axK_1.set(xlabel='Temp K',ylabel='Conductivity (Solid) W/mK',title='Smoothed Univariate Spline')

#axK_2.plot(K_temp_range,K_data,'b+',label='data')
#axK_2.plot(T_test_data,K_ispline(T_test_data),'r--',label='I-Spline')
#axK_2.set(xlabel='Temp K',ylabel='Conductivity (Solid) W/mK',title='Interpolated Univariate Spline')

#axK_3.plot(K_temp_range,K_data,'b+',label='data')
#axK_3.set(xlabel='Temp K',ylabel='Conductivity (Solid) W/mK',title='Data')
##powder
#axK_4.plot(K_temp_range,Kp_data,'b+',label='data')
#axK_4.plot(T_test_data,Kp_spline(T_test_data),'g-',label='S-Spline')
#axK_4.set(xlabel='Temp K',ylabel='Conductivity (Powder) W/mK',title='Smoothed Univariate Spline')

#axK_5.plot(K_temp_range,Kp_data,'b+',label='data')
#axK_5.plot(T_test_data,Kp_ispline(T_test_data),'r--',label='I-Spline')
#axK_5.set(xlabel='Temp K',ylabel='Conductivity (Powder) W/mK',title='Interpolated Univariate Spline')

#axK_6.plot(K_temp_range,K_data,'b+',label='data')
#axK_6.set(xlabel='Temp K',ylabel='Conductivity (Powder) W/mK',title='Data')

## show legends
#axK_1.legend()
#axK_2.legend()
#axK_3.legend()
#axK_4.legend()
#axK_5.legend()
#axK_6.legend()

#f.savefig('temp-K-predict.png')

## specific heat capacity
#f,(axC_1,axC_2,axC_3) = plt.subplots(1,3,tight_layout=True,figsize=param_fsize)
#f.suptitle("Specific Heat Capacity Spline Prediction")
#axC_1.plot(C_temp_range,C_data,'b+',label='data')
#axC_1.plot(T_test_data,C_spline(T_test_data),'g-',label='S-Spline')
#axC_1.set(xlabel='Temp K',ylabel='Specific Heat Capacity J/kg.K',title='Smoothed Univariate Spline')

#axC_2.plot(C_temp_range,C_data,'b+',label='data')
#axC_2.plot(T_test_data,C_ispline(T_test_data),'r--',label='I-Spline')
#axC_2.set(xlabel='Temp K',ylabel='Specific Heat Capacity J/kg.K',title='Interpolated Univariate Spline')

#axC_3.plot(C_temp_range,C_data,'b+',label='data')
#axC_3.set(xlabel='Temp K',ylabel='Specific Heat Capacity J/kg.K',title='Data')

## show legends
#axC_1.legend()
#axC_2.legend()
#axC_3.legend()

#f.savefig('temp-C-predict.png')

## thermal volume expansion
#f,(axE_1,axE_2,axE_3) = plt.subplots(1,3,tight_layout=True,figsize=param_fsize)
#f.suptitle("Thermal Volumetric Expansion Spline Prediction")
#axE_1.plot(Ev_temp_range,Ev_data,'b+',label='data')
#axE_1.plot(T_test_data,Ev_spline(T_test_data),'g-',label='S-Spline')
#axE_1.set(xlabel='Temperature Change K',ylabel='Volume Expansion mm^3/mm^3.C',title='Smoothed Univariate Spline')

#axE_2.plot(C_temp_range,C_data,'b+',label='data')
#axE_2.plot(T_test_data,C_ispline(T_test_data),'r--',label='I-Spline')
#axE_2.set(xlabel='Temperature Change K',ylabel='Volume Expansion mm^3/mm^3.C',title='Interpolated Univariate Spline')

#axE_3.plot(C_temp_range,C_data,'b+',label='data')
#axE_3.set(xlabel='Temperature Change K',ylabel='Volume Expansion mm^3/mm^3.C',title='Data')

## show legends
#axE_1.legend()
#axE_2.legend()
#axE_3.legend()

#f.savefig('thermal-volume-expansion-predict.png')

## solid material density
#f,(axp_1,axp_2,axp_3) = plt.subplots(1,3,tight_layout=True,figsize=param_fsize)
#f.suptitle("Density Spline Prediction")
#axp_1.plot(p_temp_range,p_data,'b+',label='data')
#axp_1.plot(T_test_data,p_spline(T_test_data),'g-',label='S-Spline')
#axp_1.set(xlabel='Temperature K',ylabel='Solid Density kg/m^3',title='Smoothed Univariate Spline')

#axp_2.plot(p_temp_range,p_data,'b+',label='data')
#axp_2.plot(T_test_data,p_ispline(T_test_data),'r--',label='I-Spline')
#axp_2.set(xlabel='Temperature K',ylabel='Solid Density kg/m^3',title='Interpolated Univariate Spline')

#axp_3.plot(p_temp_range,p_data,'b+',label='data')
#axp_3.set(xlabel='Temperature K',ylabel='Solid Density kg/m^3',title='Data')

## show legends
#axp_1.legend()
#axp_2.legend()
#axp_3.legend()

#f.savefig('solid-density-predict.png')

### thermal diffusivity
## solid material
#f,((axDs_1,axDs_2,axDs_3),(axDs_4,axDs_5,axDs_6)) = plt.subplots(2,3,tight_layout=True,figsize=param_fsize)
#f.suptitle("Thermal Diffusivity Spline Prediction")
#axDs_1.plot(T_test_data,Ds_data,'b+',label='data')
#axDs_1.plot(T_test_data,Ds_spline(T_test_data),'g-',label='S-Spline')
#axDs_1.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Solid)',title='Smoothed Univariate Spline')

#axDs_2.plot(T_test_data,Ds_data,'b+',label='data')
#axDs_2.plot(T_test_data,Ds_ispline(T_test_data),'r--',label='I-Spline')
#axDs_2.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Solid)',title='Interpolated Univariate Spline')

#axDs_3.plot(T_test_data,Ds_data,'b+',label='data')
#axDs_3.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Solid)',title='Data')

#axDs_4.plot(T_test_data,Dp_data,'b+',label='data')
#axDs_4.plot(T_test_data,Dp_spline(T_test_data),'g-',label='S-Spline')
#axDs_4.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Powder)',title='Smoothed Univariate Spline')

#axDs_5.plot(T_test_data,Dp_data,'b+',label='data')
#axDs_5.plot(T_test_data,Dp_ispline(T_test_data),'r--',label='I-Spline')
#axDs_5.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Powder)',title='Interpolated Univariate Spline')

#axDs_6.plot(T_test_data,Ds_data,'b+',label='data')
#axDs_6.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Powder)',title='Data')

## show legends
#axDs_1.legend()
#axDs_2.legend()
#axDs_3.legend()
#axDs_4.legend()
#axDs_5.legend()
#axDs_6.legend()

#f.savefig('thermal-diffusivity-predict.png')

#plt.show()

# pixel pitch
pixel_pitch = 20.0*10.0**-6 # metres
# pixel area
pixel_area = pixel_pitch**2.0 # m2

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
    """
    from scipy.constants import Stefan_Boltzmann as sigma
    from numpy import asarray
    # if emissivity is 0.0, return zeros matrix
    # avoids attempt to divide by 0
    if e==0.0:
        #print("emissivity is 0")
        return np.zeros(asarray(qr).shape,qr.dtype)
    else:
        # to do calculation
        # try-except is to manage invalid variables or divide by zero errors
        with np.errstate(divide='raise',invalid='raise'):
            try:
                #return ((asarray(T0))**4.0 - (asarray(qr)/(e*sigma)))**0.25
                return ((asarray(qr)/(e*sigma)) + (asarray(T0))**4.0 )**0.25
            except:
                #print("Catch in Qr2Temp")
                return np.zeros(asarray(qr).shape,qr.dtype)

def I2Power(I,D):
    """ Estimated power for given power density fn assuming it follows a 2D fn
    
        I : Power density matrix, W/m^2
        D : thermal diffusivity, m^2/s

        Returns estimated power to produce given power density

        Volume under a 2D gaussian is V = 2*PI*peak*std
    """
    from numpy import asarray,max,pi
    return 2*pi*max(asarray(I))*D

def Temp2IApprox(T,T0,K,D,t):
    """ Function for converting temperature to power density

        K : thermal conductivity, spline fn
        D : thermal diffusivity, m^2/s
        t : interaction time between T0 and T
        T : temperature K, numpy matrix
        T0: starter temperature K , constant

        Returns power density approximation ndarray and laser power estimate

        Approximation based on the assumption of uniform energy for the given area
    """
    # get numpy fn to interpret temperature as matrix and get pi constant
    from numpy import asarray, pi, sqrt, arange
    # try-except to manage div 0 error
    try:
        return K(asarray(T)-asarray(T0))*(T-T0)/(2*sqrt(asarray(D)*t/pi))
    # runtme warning for div 0
    except RuntimeWarning:
        return np.zeros(T.shape,dtype=T.dtype)

# fn for euclidean distance between points
def euc_dist_pair(p1,p2): return sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def max_dist(r_c):
    """ Find the pair of r,c points that are furthest apart

        r_c : array of rc pairs

        Return max distance
    """
    from itertools import combinations
    max_sq_dist = 0
    # for each combination of points
    for pair_pair in combinations(r_c,2):
        # calculate euclidean distance between points
        dist = euc_dist_pair(*pair_pair)
        # if greater than max distance
        if dist>max_sq_dist:
            # update distance
            max_sq_dist = dist
    # return the maximum distance and the pair of points that are furtherst apart
    return max_sq_dist

def estR0(I,method):
    """ Estimate laser beam diameter from power density

        I : Power density matrix
        method : method for estimating r0, options="e2","FWHM","4sigma"
        Returns laser radius in m according to different methods

        1/e^2 : Distance between two points where power density is 1/e^2 od peak
        FWHM  : Distance between two points where power density if half
        4DSigma: 4 times the standard deviation of the power density matrix
        
        Return all three as 1/e^2 , FWHM, 4DSigma
    """
    from numpy import where, max, isclose, full
    from math import exp
    #from itertools import combinations
    if method=="e2":
        # find indicies where power is around 1/e^2 power
        # isclose is being used to account for inaccuracies in conversion and estimation
        # returns an array of row and col indicies (r_idx, c_idx)
        rc = where(where(isclose(I,full(I.shape[:2],(max(I)*(1.0/exp(2)))))))
        r,c = rc
        from scipy.spatial.distance import pdist, squareform
        if r.shape[0]>2:
            # euclidean distance between points, condensed matrix, 1D
            dist_dense = pdist(rc)
            # convert to sparse matrix to make indexing easier, 2D
            dist_sparse = squareform(dist_dense)
            return max(dist_dense)*pixel_pitch
        else:
            return 0.0

    elif method=="FWHM":
        # find indicies where power is around 1/e^2 power
        # isclose is being used to account for inaccuracies in conversion and estimation
        # returns an array of row and col indicies (r_idx, c_idx)
        r,c = where(isclose(I,full(I.shape[:2],max(I)/2.0)))
        #fw_peaki_r,fw_peaki_c = where(I==(max(I)/2.0))
        if r.shape[1]>0:
            # turn array of two arrays to an array of pairs
            rc = np.concatenate((r,c),1)
            # get maximum distance between points
            return max_dist(rc)
        else:
            return 0.0
    elif method=="4sigma":
        # 4 times the standard deviation
        # orig value in pixels. converted to m using pixel pitch
        return I.std()*4.0*pixel_pitch

## estimating reflectivity from 2d gaussian
def Temp2Ref(T,T0,Ks,P,v,D,PP):
    """ Estimating reflectivity from temperature.
        
        T    : temperature matrix (K)
        T0   : starter temperature (K)
        Ks   : thermal conductivity spline
        P    : laser power (W)
        v    : track velocity (ms-1)
        D    : thermal diffusivity constant for current frame (m^2/s)
        PP   : pixel pitch (m)

        Returns reflectivity matrix

        Based on the equation for temperature induced by a laser moving along a 
        1D track in the x-direction
    """
    from numpy import asarray,pi,exp,max,where,linspace,mean,argmax,unravel_index
    from scipy.spatial.distance import cdist
    # find where peak occurs aka mean
    # pixel idx
    idx_dense = argmax(T,axis=None)
    if idx_dense>0:
        # get idx of mean/peak
        x0,y0 = unravel_index(idx_dense,T.shape)
        # get size of matrix
        rows,cols = T.shape[:2]
        # create x,y ccoordinates, m
        x = linspace(0,(rows-1)*PP,rows)
        y = linspace(0,(cols-1)*PP,cols)
        # create 2d coordinate matrix
        X, Y = meshgrid(x,y)
        # get x distance from mean
        X_dist = X-x0
        # convert coordinate matrix to 1D
        X = X.reshape((prod(X.shape),))
        Y = Y.reshape((prod(Y.shape),))
        # create list of coordinate pairs
        coord = asarray(zip(X,Y))
        # compute pairwise distance between coordinate pairs
        # get slice for distances from mean and reshape to matrix
        # default is euclidean metric
        r = cdist(coord,coord)[:,x0*y0].reshape(T.shape)
        # estimate reflectivity
        R = 1-(2*pi*Ks(T-T0)*(T-asarray(T0))/P)*exp(v*(r+X_dist)/(2*D))
        # return max and mean, don't know which one is best
        return max(R),mean(R)
    else:
        return 0.0,0.0

def estR0_m(I,PP):
    """ Estimating laser radius based on second moment of intensity distribution 
    
        I : Power intensity matrix
        PP: pixel pitch (m)

        Returns estimated laser radius based off second movement of data
    """
    # https://www.rp-photonics.com/beam_radius.html
    from numpy import asarray,sqrt,sum,linspace,meshgrid,argmax,unravel_index
    rows,cols = asarray(I).shape[:2]
    # create x,y ccoordinates, m
    x = linspace(0,(rows-1)*PP,rows)
    y = linspace(0,(cols-1)*PP,cols)
    # create 2d coordinate matrix
    X,_ = meshgrid(x,y)
    # find where peak occurs
    pk_didx = argmax(I)
    # if found
    if pk_didx>0:
        # convert to x,y coordinate
        x0,_ = unravel_index(pk_didx,I.shape)
        # the subtaction is required as moment distance has to be relative to peak
        # calculate second moment according to source
        return 2.0*sqrt(sum(((X-x0)**2)*I)/sum(I))
    else:
        return 0.0
    

print("Processing data")
### SETTING AXIS PARAMETER TO Z MEANS THE OPERATION IS PERFORMED ON EACH ELEMENT OF THE MATRIX ###
### RESULT IS A 128x128
# get peak temperature
#Temp_peak_ep = np.max(Tsurf_ep,axis=2)
#Temp_peak_ess = np.max(Tsurf_ess,axis=2)
#print("T Peak Mat Size: ",Temp_peak_ep.shape)

# thermal diffusivity, m^2/s
# population standard deviation
#D_ess = np.std(Tsurf_ess,axis=2)*10**-6
#D_ep = np.std(Tsurf_ep,axis=2)*10**-6
#print("Thermal Diff Size: ",D_ess.shape)
#print("Estimating temp peak and D")

# peak velocity, read from xml graph
# used in reflectivity estimate
v_peak = 0.004*10**3 # m/s

## for each emissivity value
# arrays for data, number of e x num_frames
e_res = 0.1 # resolution of emissivity array
e_range = np.arange(0,1.0,e_res)
num_e = e_range.shape[0]

## cold start data
# power matrix
Power = np.zeros((num_e,num_frames))
# reflectivity
#R = np.zeros((num_e,num_frames))
# thermal diffusivity matrix
# same for both
D = np.zeros((num_e,num_frames))
D_mean = np.zeros((num_e,num_frames))
# peak temperature
Temp_peak = np.zeros((num_e,num_frames))
# laser radius estimate
r0 = np.zeros((num_e,num_frames))
r0_m2 = np.zeros((num_e,num_frames))
# surface temperature matrix
Tsurf = np.zeros((rows,cols,num_frames),dtype=frames_set_array.dtype)

## relative to prev frame
# power matrix
Power_rel = np.zeros((num_e,num_frames))
# reflectivity
#R_rel = np.zeros((num_e,num_frames))
# thermal diffusivity matrix
# same for both
D_rel = np.zeros((num_e,num_frames))
D_mean_rel = np.zeros((num_e,num_frames))
# peak temperature
Temp_peak_rel = np.zeros((num_e,num_frames))
# laser radius estimate
r0_rel = np.zeros((num_e,num_frames))
r0_m2_rel = np.zeros((num_e,num_frames))
# surface temperature matrix
Tsurf_rel = np.zeros((rows,cols,num_frames),dtype=frames_set_array.dtype)

## Setting which splines to use
usePowderEst = False
useExtrapolatedK = False
useExtrapolatedD = True
# if using powder splines
if usePowderEst:
    if useExtrapolatedK:
        print("Using extrapolated powder K spline")
        K_fn = Kp_spline
        K_name = "Kp-spline"
    else:
        print("Using interpolated powder K spline")
        K_fn = Kp_ispline
        K_name = "Kp-ispline"

    if useExtrapolatedD:
        print("Using extrapolated powder D spline")
        D_fn = Dp_spline
        D_name = "Dp-spline"
    else:
        print("Using interpolated powder D spline")
        D_fn = Dp_ispline
        D_name = "Dp-ispline"
# if using solid splines
else:
    if useExtrapolatedK:
        print("Using extrapolated solid K spline")
        K_fn = K_spline
        K_name = "Ks-spline"
    else:
        print("Using interpolated solid K spline")
        K_fn = K_ispline
        K_name = "Ks-ispline"

    if useExtrapolatedD:
        print("Using extrapolated solid D spline")
        D_fn = Ds_spline
        D_name = "Ds-spline"
    else:
        print("Using interpolated solid D spline")
        D_fn = Ds_ispline
        D_name = "Ds-ispline"


for ei,e in enumerate(e_range):
    print("e: ", e)
    ##log.writeToLog("Using e={:.2f}".format(e))
    # surface temp data for cold start
    #log.writeToLog("Creating surface temperature matrix for cold start")
    Tsurf = np.apply_along_axis(Qr2Temp,2,frames_set_array,e,T0)
    
    if np.any(np.isnan(x) for x in Tsurf.flatten()):
        print("Found nan in cold start temperature matrix. Check fns")
    
    for f in np.arange(0,num_frames-1,1):
        # if first frame, use T0 as previous temperature
        #log.writeToLog("Creating surface temperature matrix for relative frame")
        if f==0:
            Tsurf_rel[:,:,f] = Qr2Temp(frames_set_array[:,:,f],e,T0)
        else:
            Tsurf_rel[:,:,f] = Qr2Temp(frames_set_array[:,:,f],e,Tsurf_rel[:,:,f-1])

        if np.isnan(np.min(Tsurf_rel[:,:,f])):
            print("Found nan in relative time temperature matrix. Check fns")

        #log.writeToLog("Storing temperature peaks")
        # temperature peak
        Temp_peak_rel[ei,f]=np.max(Tsurf_rel[:,:,f])
        Temp_peak[ei,f]=np.max(Tsurf[:,:,f])
        # thermal diffusivity using spline
        # storing peak value
        #log.writeToLog("Getting thermal diffusivity")
        D[ei,f]=np.max(D_fn(Tsurf[:,:,f]))
        D_mean[ei,f]=np.mean(D_fn(Tsurf[:,:,f]))
        D_rel[ei,f]=np.max(D_fn(Tsurf_rel[:,:,f]))
        D_mean_rel[ei,f]=np.mean(D_fn(Tsurf_rel[:,:,f]))
        ## laser power estimate
        ## can be vectorized, req more RAM/memory, MemoryError
        # cold start
        #log.writeToLog("Approximating power for cold start")
        I = Temp2IApprox(Tsurf[:,:,f],T0,K_fn,D[ei,f],tc*float(f))
        Power[ei,f]=(I2Power(I,D[ei,f]))
        #log.writeToLog("Approximating laser radius for cold start")
        # estimate laser radius using 1/e^2 method
        r0[ei,f]=(estR0(I,"e2"))
        # estimate laser radius using 2nd moment
        r0_m2[ei,f] = estR0_m(I,pixel_pitch)

        # relative frame
        #log.writeToLog("Approximating power for relative frame")
        I = Temp2IApprox(Tsurf[:,:,f],T0,K_fn,D_rel[ei,f],tc)
        Power_rel[ei,f]=(I2Power(I,D_rel[ei,f]))
        # estimate laser radius using 1/e^2 method
        #log.writeToLog("Approximating laser radius for relative frame")
        r0_rel[ei,f] = estR0(I,"e2")
        r0_m2_rel[ei,f] = estR0_m(I,pixel_pitch)
        # estimate max reflectivity
        #R[ei,f] = Temp2Ref(Tsurf,T0,K_ispline,Power[ei,f],v_peak,D[ei,f],pixel_pitch)[0]


# set figure size to scaled version of default fig size
scale_factor = 2
scaled_size = [f*scale_factor for f in plt.rcParams["figure.figsize"]]
#log.writeToLog("Plotting results")
print("Plotting relative frame")
f,(ax1,ax2) = plt.subplots(1,2,tight_layout=True,figsize=scaled_size)
ax1.set(xlabel='Frame Idx',ylabel='Thermal Diffusivity (m^2/s)',title='Max Thermal Diffusivity, Rel. Time')
ax2.set(xlabel='Frame Idx',ylabel='Thermal Diffusivity (m^2/s)',title='Mean Thermal Diffusivity, Rel. Time')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax1.plot(D_rel[i,:],label="e={:.2f}".format(e_range[i]))
    ax2.plot(D_mean_rel[i,:],label="e={:.2f}".format(e_range[i]))

#ax1.ticklabel_format(axis='y', style='sci',scilimits=(-7,-5))
#ax2.ticklabel_format(axis='y', style='sci',scilimits=(-7,-5))
ax1.legend()
ax2.legend()
f.savefig("thermal-diff-e-range-rel-time-{0}-{1}.png".format(K_name,D_name))

f,ax = plt.subplots()
f.suptitle("Temperature Peak for Range of e, Relative Time")
ax.set(xlabel='Frame Idx',ylabel='Peak Temp (K)')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax.plot(Temp_peak_rel[i,:],label="e={:.2f}".format(e_range[i]))

ax.legend()
f.savefig("temp-peak-e-range-rel-time-{0}-{1}.png".format(K_name,D_name))

## impact of emmisivity on temperature
f,(ax1,ax2) = plt.subplots(1,2)
f.suptitle("Impact of Emissivity on Peak Temperature, Relative Time")
# get the min and max pk temperature from each run
pk_min = np.min(Temp_peak_rel,axis=1)
pk_max = np.max(Temp_peak_rel,axis=1)
ax1.plot(e_range,pk_min,label="Min. Pk Temp")
ax1.set(xlabel='Emissivity',ylabel='Peak Temp (K)',title='Minimum Peak Temperature')
ax2.plot(e_range,pk_max,label="Max. Pk Temp")
ax2.set(xlabel='Emissivity',ylabel='Peak Temp (K)',title='Maximum Peak Temperature')
# add legend
ax1.legend()
ax2.legend()

f.savefig("temp-peak-eimpact-rel-time-{0}-{1}.png".format(K_name,D_name))

f,ax = plt.subplots()
f.suptitle("Change in Laser Power for Range of e, Relative Time")
ax.set(xlabel='Frame Idx',ylabel='Laser Power (W)')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax.plot(Power_rel[i,:],label="e={:.2f}".format(e_range[i]))
ax.legend()
f.savefig("laser-power-e-range-rel-time-{0}-{1}.png".format(K_name,D_name))

f,ax = plt.subplots()
f.suptitle("Laser Radius Estimate (1/e2) method for Range of e, Relative Time")
ax.set(xlabel='Frame Idx',ylabel='Laser Radius (m)')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax.plot(r0_rel[i,:],label="e={:.2f}".format(e_range[i]))
ax.legend()
f.savefig("laser-radius-e-2-e-range-rel-time-{0}-{1}.png".format(K_name,D_name))

f,ax = plt.subplots()
f.suptitle("Laser Radius Estimate (m2) for Range of e, Relative Time")
ax.set(xlabel='Frame Idx',ylabel='Laser Radius (m)')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax.plot(r0_m2_rel[i,:],label="e={:.2f}".format(e_range[i]))
ax.legend()
f.savefig("laser-radius-m2-e-range-rel-time-{0}-{1}.png".format(K_name,D_name))

#f,ax = plt.subplots()
#f.suptitle("Reflectivity for Range of e")
#ax.set(xlabel='Frame Idx',ylabel='Reflectivity')
## for each e value
#for i in range(num_e):
#    # iterate through each plot adding it to axis
#    ax.plot(R[i,:],label="e={:.2f}".format(e_range[i]))
#ax.legend()
#f.savefig("reflectivity-max-e-range-rel-time.png")

###########################################################
print("Plotting cold start data")
f,(ax1,ax2) = plt.subplots(1,2,tight_layout=True,figsize=scaled_size)
ax1.set(xlabel='Frame Idx',ylabel='Thermal Diffusivity (m^2/s)',title='Max Thermal Diffusivity, Cold Start')
ax2.set(xlabel='Frame Idx',ylabel='Thermal Diffusivity (m^2/s)',title='Mean Thermal Diffusivity, Cold Start')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax1.plot(D[i,:],label="e={:.2f}".format(e_range[i]))
    ax2.plot(D_mean[i,:],label="e={:.2f}".format(e_range[i]))

#ax1.ticklabel_format(axis='y', style='sci',scilimits=(-7,-5))
#ax2.ticklabel_format(axis='y', style='sci',scilimits=(-7,-5))
ax1.legend()
ax2.legend()
f.savefig("thermal-diff-e-range-{0}-{1}.png".format(K_name,D_name))

f,ax = plt.subplots()
f.suptitle("Temperature Peak for Range of e, Cold Start")
ax.set(xlabel='Frame Idx',ylabel='Peak Temp (K)')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax.plot(Temp_peak[i,:],label="e={:.2f}".format(e_range[i]))

ax.legend()
f.savefig("temp-peak-e-range-{0}-{1}.png".format(K_name,D_name))

## impact of emmisivity on temperature
f,(ax1,ax2) = plt.subplots(1,2)
f.suptitle("Impact of Emissivity on Peak Temperature, Cold Start")
# get the min and max pk temperature from each run
pk_min = np.min(Temp_peak,axis=1)
pk_max = np.max(Temp_peak,axis=1)
ax1.plot(e_range,pk_min,label="Min. Pk Temp")
ax1.set(xlabel='Emissivity',ylabel='Peak Temp (K)',title='Minimum Peak Temperature')
ax2.plot(e_range,pk_max,label="Max. Pk Temp")
ax2.set(xlabel='Emissivity',ylabel='Peak Temp (K)',title='Maximum Peak Temperature')
# add legend
ax1.legend()
ax2.legend()

f.savefig("temp-peak-eimpact-{0}-{1}.png".format(K_name,D_name))

f,ax = plt.subplots()
f.suptitle("Laser Power for Range of e, Cold Start")
ax.set(xlabel='Frame Idx',ylabel='Laser Power (W)')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax.plot(Power[i,:],label="e={:.2f}".format(e_range[i]))
ax.legend()
f.savefig("laser-power-e-range-{0}-{1}.png".format(K_name,D_name))

f,ax = plt.subplots()
f.suptitle("Laser Radius Estimate (1/e2) for Range of e, Cold Start")
ax.set(xlabel='Frame Idx',ylabel='Laser Radius (m)')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax.plot(r0[i,:],label="e={:.2f}".format(e_range[i]))
ax.legend()
f.savefig("laser-radius-e-2-e-range-{0}-{1}.png".format(K_name,D_name))

f,ax = plt.subplots()
f.suptitle("Laser Radius Estimate (m2) for Range of e, Cold Start")
ax.set(xlabel='Frame Idx',ylabel='Laser Radius (m)')
# for each e value
for i in range(num_e):
    # iterate through each plot adding it to axis
    ax.plot(r0_m2[i,:],label="e={:.2f}".format(e_range[i]))
ax.legend()
f.savefig("laser-radius-m2-e-range-{0}-{1}.png".format(K_name,D_name))

#f,ax = plt.subplots()
#f.suptitle("Reflectivity for Range of e")
#ax.set(xlabel='Frame Idx',ylabel='Reflectivity')
## for each e value
#for i in range(num_e):
#    # iterate through each plot adding it to axis
#    ax.plot(R[i,:],label="e={:.2f}".format(e_range[i]))
#ax.legend()
#f.savefig("reflectivity-max-e-range-rel-time.png")

###########################################################
###########################################################

print("Finished")