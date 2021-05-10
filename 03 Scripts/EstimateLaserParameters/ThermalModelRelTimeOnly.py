from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from matplotlib.offsetbox import AnchoredText
import os
from TSLoggerPy.TSLoggerPy import TSLoggerPy as logger

#log = logger()

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
print("Original data type: ",frames_set_array.dtype)

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
#Qr_mat = frames_set_array.astype(dtype=np.float16)
print("New type: ",frames_set_array.dtype)

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
##param_fsize = (12.0, 8.0)
## #thermal conductivity
##print("Plotting parameter data")
##f,((axK_1,axK_2,axK_3),(axK_4,axK_5,axK_6)) = plt.subplots(2,3,constrained_layout=True,figsize=param_fsize)
##f.suptitle("Thermal Conductivity Spline Prediction")
## #solid
##axK_1.plot(K_temp_range,K_data,'b+',label='data')
##axK_1.plot(T_test_data,K_spline(T_test_data),'g-',label='S-Spline')
##axK_1.set(xlabel='Temp K',ylabel='Conductivity (Solid) W/mK',title='Smoothed Univariate Spline')
##
##axK_2.plot(K_temp_range,K_data,'b+',label='data')
##axK_2.plot(T_test_data,K_ispline(T_test_data),'r--',label='I-Spline')
##axK_2.set(xlabel='Temp K',ylabel='Conductivity (Solid) W/mK',title='Interpolated Univariate Spline')
##
##axK_3.plot(K_temp_range,K_data,'b+',label='data')
##axK_3.set(xlabel='Temp K',ylabel='Conductivity (Solid) W/mK',title='Data')
###powder
##axK_4.plot(K_temp_range,Kp_data,'b+',label='data')
##axK_4.plot(T_test_data,Kp_spline(T_test_data),'g-',label='S-Spline')
##axK_4.set(xlabel='Temp K',ylabel='Conductivity (Powder) W/mK',title='Smoothed Univariate Spline')
##
##axK_5.plot(K_temp_range,Kp_data,'b+',label='data')
##axK_5.plot(T_test_data,Kp_ispline(T_test_data),'r--',label='I-Spline')
##axK_5.set(xlabel='Temp K',ylabel='Conductivity (Powder) W/mK',title='Interpolated Univariate Spline')
##
##axK_6.plot(K_temp_range,K_data,'b+',label='data')
##axK_6.set(xlabel='Temp K',ylabel='Conductivity (Powder) W/mK',title='Data')
##
## #show legends
##axK_1.legend()
##axK_2.legend()
##axK_3.legend()
##axK_4.legend()
##axK_5.legend()
##axK_6.legend()
##
##f.savefig('temp-K-predict.png')
##
## #specific heat capacity
##f,(axC_1,axC_2,axC_3) = plt.subplots(1,3,constrained_layout=True,figsize=param_fsize)
##f.suptitle("Specific Heat Capacity Spline Prediction")
##axC_1.plot(C_temp_range,C_data,'b+',label='data')
##axC_1.plot(T_test_data,C_spline(T_test_data),'g-',label='S-Spline')
##axC_1.set(xlabel='Temp K',ylabel='Specific Heat Capacity J/kg.K',title='Smoothed Univariate Spline')
##
##axC_2.plot(C_temp_range,C_data,'b+',label='data')
##axC_2.plot(T_test_data,C_ispline(T_test_data),'r--',label='I-Spline')
##axC_2.set(xlabel='Temp K',ylabel='Specific Heat Capacity J/kg.K',title='Interpolated Univariate Spline')
##
##axC_3.plot(C_temp_range,C_data,'b+',label='data')
##axC_3.set(xlabel='Temp K',ylabel='Specific Heat Capacity J/kg.K',title='Data')
##
## #show legends
##axC_1.legend()
##axC_2.legend()
##axC_3.legend()
##
##f.savefig('temp-C-predict.png')
##
## #thermal volume expansion
##f,(axE_1,axE_2,axE_3) = plt.subplots(1,3,constrained_layout=True,figsize=param_fsize)
##f.suptitle("Thermal Volumetric Expansion Spline Prediction")
##axE_1.plot(Ev_temp_range,Ev_data,'b+',label='data')
##axE_1.plot(T_test_data,Ev_spline(T_test_data),'g-',label='S-Spline')
##axE_1.set(xlabel='Temperature Change K',ylabel='Volume Expansion mm^3/mm^3.C',title='Smoothed Univariate Spline')
##
##axE_2.plot(C_temp_range,C_data,'b+',label='data')
##axE_2.plot(T_test_data,C_ispline(T_test_data),'r--',label='I-Spline')
##axE_2.set(xlabel='Temperature Change K',ylabel='Volume Expansion mm^3/mm^3.C',title='Interpolated Univariate Spline')
##
##axE_3.plot(C_temp_range,C_data,'b+',label='data')
##axE_3.set(xlabel='Temperature Change K',ylabel='Volume Expansion mm^3/mm^3.C',title='Data')
##
## #show legends
##axE_1.legend()
##axE_2.legend()
##axE_3.legend()
##
##f.savefig('thermal-volume-expansion-predict.png')
##
## #solid material density
##f,(axp_1,axp_2,axp_3) = plt.subplots(1,3,constrained_layout=True,figsize=param_fsize)
##f.suptitle("Density Spline Prediction")
##axp_1.plot(p_temp_range,p_data,'b+',label='data')
##axp_1.plot(T_test_data,p_spline(T_test_data),'g-',label='S-Spline')
##axp_1.set(xlabel='Temperature K',ylabel='Solid Density kg/m^3',title='Smoothed Univariate Spline')
##
##axp_2.plot(p_temp_range,p_data,'b+',label='data')
##axp_2.plot(T_test_data,p_ispline(T_test_data),'r--',label='I-Spline')
##axp_2.set(xlabel='Temperature K',ylabel='Solid Density kg/m^3',title='Interpolated Univariate Spline')
##
##axp_3.plot(p_temp_range,p_data,'b+',label='data')
##axp_3.set(xlabel='Temperature K',ylabel='Solid Density kg/m^3',title='Data')
##
## #show legends
##axp_1.legend()
##axp_2.legend()
##axp_3.legend()
##
##f.savefig('solid-density-predict.png')
##
### thermal diffusivity
## #solid material
##f,((axDs_1,axDs_2,axDs_3),(axDs_4,axDs_5,axDs_6)) = plt.subplots(2,3,constrained_layout=True,figsize=param_fsize)
##f.suptitle("Thermal Diffusivity Spline Prediction")
##axDs_1.plot(T_test_data,Ds_data,'b+',label='data')
##axDs_1.plot(T_test_data,Ds_spline(T_test_data),'g-',label='S-Spline')
##axDs_1.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Solid)',title='Smoothed Univariate Spline')
##
##axDs_2.plot(T_test_data,Ds_data,'b+',label='data')
##axDs_2.plot(T_test_data,Ds_ispline(T_test_data),'r--',label='I-Spline')
##axDs_2.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Solid)',title='Interpolated Univariate Spline')
##
##axDs_3.plot(T_test_data,Ds_data,'b+',label='data')
##axDs_3.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Solid)',title='Data')
##
##axDs_4.plot(T_test_data,Dp_data,'b+',label='data')
##axDs_4.plot(T_test_data,Dp_spline(T_test_data),'g-',label='S-Spline')
##axDs_4.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Powder)',title='Smoothed Univariate Spline')
##
##axDs_5.plot(T_test_data,Dp_data,'b+',label='data')
##axDs_5.plot(T_test_data,Dp_ispline(T_test_data),'r--',label='I-Spline')
##axDs_5.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Powder)',title='Interpolated Univariate Spline')
##
##axDs_6.plot(T_test_data,Ds_data,'b+',label='data')
##axDs_6.set(xlabel='Temperature K',ylabel='Thermal Diffusivity (Powder)',title='Data')
##
## #show legends
##axDs_1.legend()
##axDs_2.legend()
##axDs_3.legend()
##axDs_4.legend()
##axDs_5.legend()
##axDs_6.legend()
##
##f.savefig('thermal-diffusivity-predict.png')

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
        a = qr*div
        # T0^4
        #print("T0:",T0)
        b = asarray(T0,dtype=np.float32)**4.0
        #print("To^4:",b)
        # (qr/(e*sigma)) + T0^4
        c = a+b
        return (c)**0.25

    
def I2Power(I):
    """ Estimated power for given power density fn assuming it follows a 2D fn
    
        I : Power density matrix, W/m^2
        D : thermal diffusivity, m^2/s

        Returns estimated power to produce given power density

        Volume under a 2D gaussian is V = 2*PI*peak*std^2
    """
    from numpy import asarray,pi,std
    from numpy import max as nmax
    # caculate peak
    Ipk = nmax(asarray(I,dtype=np.float32))
    # calculate standard deviation if I
    Istd = std(I)
    # calculate volume under bivariate gaussian
    return 2.0*pi*Ipk*(Istd**2.0)

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
    a = (D*t)/np.pi
    # result of sqrt can be +/-
    # power density cannot be negative 
    b = (2.0*np.sqrt(a))
    temp = K*Tdiff
    # K*(T-T0)/(2*sqrt(Dt/pi))
    return abs(temp/b)
    

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
    from numpy import asarray,pi,exp,max,where,linspace,mean,argmax,unravel_index,prod
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
    x = linspace(0,(rows-1),rows)
    y = linspace(0,(cols-1),cols)
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
        return 2.0*sqrt(sum(((X-x0)**2)*I)/sum(I))*PP
    else:
        return 0.0

def estR0Tree(I,PP):
    """ Estimate laser radius using K-D tree

        I : power density matrix
        PP : pixel pitch (m^2)

        Calculates w0 by constructing a k-d tree of points where
        power density drops below 1/e^2 of peak and findind the
        closest point

        Returns closest distance converted to physical space using pixel
        pitch (m^2) and estimate of laser power (W)
    """
    from scipy.spatial import cKDTree
    from numpy import asarray,array,where,exp
    from numpy import max as nmax
    # get max of power density function
    #print("\t=====In of fn=====")
    Ipk = nmax(I)
    #print("I shape:",I.shape)
    #print("I:",I)
    #print("Ipk:",Ipk)
    # find where peak is
    rpk,cpk = where(I==Ipk)
    #print("Rpk:",rpk)
    #print("Cpk:",cpk)
    # calculate 1/e^2 of peak
    Ie2 = Ipk*(1.0/(exp(1.0)**2.0))
    #print("Ie2:",Ie2)
    # find where power drops below Ie2
    re2,ce2 = where(I<=Ie2)
    #print("Re2:",re2)
    #print("Ce2:",ce2)
    # if no data points are found
    if re2.shape[0]==0:
        # return 0.0 for r0 and power as neither can be calculated
        return 0.0,0.0
    else:
        # construct array of coordinate pairs
        C = [(r,c) for r,c in zip(re2,ce2)]
        # convert list of coordinate pairs to numpy array
        C = array(C)
        #print(C.shape)
        # construct k-d tree using coordinate pairs
        # tree at node k shows euclidean distance between points and 
        # query tree for closest distance and point to peak
        # just get first distance
        dist = cKDTree(C).query(np.array([[rpk[0],cpk[0]]]))[0][0]
        # scale distance to physical space from pixel
        dist *= PP
        # estimate power based off model for He-Ne
        P = (Ipk*np.pi*((dist**2)))/2.0
        # return distance converted from pixels to physical distance
        return dist,P

def estPArea(I,PP):
    """
        Estimate laser power by estimating area of effect by KDTree

        I : Power density matrix (W/pixel)
        PP : pixel pitch (m^2)

        Constructs a KD-tree of the non-zero power denity matrix values
        and queries it to find the furthest point from peak. The distance
        is taken to be the radius of the circle and estimates the area as
        a circle. The non-zero values are then summed and multiplied
        by area estimate to get estimate of power

        Returns power estimate or 0.0 if:
            - Can't find peak power density
            - Can't find non-zero power density points
            - Query of KD Tree returns nothing
    """
    import cv2
    from numpy import where,asarray,array,pi,inf,isinf,uint8,argmax
    from numpy import max as nmax
    from numpy import sum as nsum
    from scipy.spatial import cKDTree
    # find max
    Ipk = nmax(I)
    # convert power density matrix to image, BGR
    I_img = cv2.cvtColor(uint8(I),cv2.COLOR_GRAY2BGR)
    # find where max is
    rpk,cpk = where(I==Ipk)
    # if it fails to find the peak for some reason, return 0.0
    if rpk.shape[0]>0:
        # find where active data points are
        r,c = where(I>0.0)
        # if there are no non-zero data points return 0.0
        # as we can't determine the radius
        if r.shape[0]>0:
            # construct rc coordinate pairs
            C = [(rr,cc) for rr,cc in zip(r,c)]
            # convert to array
            C = array(C)
            # get shape of power density
            rows,cols = I.shape
            # get the distances of points from peak
            # return (rows*cols)/4 neighbours
            
            # limit based on visual inspection of data
            # first distance should be furthest
            dists,idxs = cKDTree(C).query(np.array([[rpk[0],cpk[0]]]),(rows*cols))
            # check if any distances were returned
            if dists.shape[0]>0:
                # replace infs with 0.0
                # returns inf when it's run out of neighbours?
                if isinf(dists).any():
                    dists[dists==inf]=0.0

                # filter out the points whose distances are further than (rows*cols)/2.0 away
                # value based off visual inspection of data
                dists[dists>((rows*cols)/4.0)]=0.0
                
                # find where max distance is
                max_dist_loc = argmax(dists)
                # get r,c vals
                rmax,cmax = C[max_dist_loc,:]
                # triangle size
                tri_size = 2
                tri_pts = array([[(rmax-tri_size,cmax),(rmax+tri_size,cmax-tri_size),(rmax+tri_size,cmax+tri_size)]],dtype='int32')
                # correct any negative references
                tri_pts[tri_pts<0]=0
                # draw filled triangle with the location of furthest point in the centre of the triangle
                cv2.fillPoly(I_img,tri_pts,color=(255,0,0))

                # draw filled triangle with location of the peak in the centre of the triangle
                tri_pts = array([[(rpk[0]-1,cpk[0]),(rpk[0]+1,cpk[0]-1),(rpk[0]+1,cpk[0]+1)]],dtype='int32')
                cv2.fillPoly(I_img,tri_pts,color=(0,255,0))

                # draw line between two points
                cv2.line(I_img,(rpk[0],cpk[0]),(rmax,cmax),(0,0,255),1)
                
                # find furthest distance
                # double idx is bc dists shape is [1,(rows*cols)]
                max_dist = dists[0,max_dist_loc]*PP
                # estimated area of effect
                # area of a circle
                area = pi*max_dist**2.0
                # sum non-zero power density (W/m^2) and times by area to get power
                # power is the same over cross section of gaussian curve
                return I_img,nsum(I[r,c])*area
            else:
                print("No results from KD query")
                return I_img,0.0
        else:
            print("Cannot find non-zero values")
            return I_img,0.0
    else:
        print("Cannot find peak")
        return I_img,0.0

def estPArea_im(I,PP):
    """ Estimating Laser Power by Area using Contours to estimate Area

        I : Power density image (W/pixel)
        PP : Pixel pitch (m)

        Uses Chain Simple Approximation to find contours

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
    thresh_img = cv2.threshold(I_img,0,255,cv2.THRESH_OTSU)[1]
    # find contours, returne contours and hierarchy
    ct=cv2.findContours(thresh_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    if len(ct)>0:
        # sort contours by area in ascending order
        ct = sorted(ct,key=lambda x : cv2.contourArea(x))
        ct_idx = 0
        # if the largest contour is likely the outer edge of the image
        if cv2.contourArea(ct[-1]) > 0.9*(I.shape[0]*I.shape[1]):
            # use next largest
            largest_area = cv2.contourArea(ct[-2])
            ct_idx = -2
        else:
            largest_area = cv2.contourArea(ct[-1])
            ct_idx = -1
        #create mask
        mask = zeros_like(I)
        # draw contour of interest on mask filled
        cv2.drawContours(mask,ct,ct_idx,255,-1) 
        #print(largest_area,img_area)
        # sum the energy densities and multiply by area to get total energy
        # only sums the places where mask is 255
        return thresh_img,mask,nsum(I[where(mask==255)])*largest_area*(PP**2)
    else:
        return thresh_img,zeros_like(I_img),0.0

def estPArea_im2(I,PP):
    """ Estimating Laser Power by Area using Contours to estimate Area

        I : Power density image (W/pixel)
        PP : Pixel pitch (m)

        Uses no approximation to find contours

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
    thresh_img = cv2.threshold(I_img,0,255,cv2.THRESH_OTSU)[1]
    # find contours, returne contours and hierarchy
    ct=cv2.findContours(thresh_img,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
    if len(ct)>0:
        # sort contours by area in ascending order
        ct = sorted(ct,key=lambda x : cv2.contourArea(x))
        ct_idx = 0
        # if the largest contour is likely the outer edge of the image
        if cv2.contourArea(ct[-1]) > 0.9*(I.shape[0]*I.shape[1]):
            # use next largest
            largest_area = cv2.contourArea(ct[-2])
            ct_idx = -2
        else:
            largest_area = cv2.contourArea(ct[-1])
            ct_idx = -1
        #create mask
        mask = zeros_like(I)
        # draw contour of interest on mask filled
        cv2.drawContours(mask,ct,ct_idx,255,-1) 
        #print(largest_area,img_area)
        # sum the energy densities and multiply by area to get total energy
        # only sums the places where mask is 255
        return thresh_img,mask,nsum(I[where(mask==255)])*largest_area*(PP**2)
    else:
        return thresh_img,zeros_like(I_img),0.0

def estPArea_Canny(I,PP):
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
    low_threshold = 100
    ratio = 3
    # apply blur to image
    I_blur = cv2.blur(I_img,(3,3))
    # apply canny edge detection to create mask
    edges = cv2.Canny(I_blur,low_threshold,low_threshold*ratio,3)
    # find contours, returne contours and hierarchy
    ct=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)[0]
    if len(ct)>0:
        # sort contours by area in ascending order
        ct = sorted(ct,key=lambda x : cv2.contourArea(x))
        ct_idx = 0
        # if the largest contour is likely the outer edge of the image
        if cv2.contourArea(ct[-1]) > 0.9*(I.shape[0]*I.shape[1]):
            # use next largest
            largest_area = cv2.contourArea(ct[-2])
            ct_idx = -2
        else:
            largest_area = cv2.contourArea(ct[-1])
            ct_idx = -1
        #create mask
        mask = zeros_like(I_img)
        # draw contour of interest on mask filled
        cv2.drawContours(mask,ct,ct_idx,255,-1) 
        #print(largest_area,img_area)
        # sum the energy densities and multiply by area to get total energy
        # only sums the places where mask is 255
        return edges,mask,nsum(I[where(mask==255)])*largest_area*(PP**2)
    else:
        return edges,zeros_like(I_img),0.0
        
print("Processing data")

# peak velocity, read from xml graph
# used in reflectivity estimate
v_peak = 0.004*10**3 # m/s

## for each emissivity value
# arrays for data, number of e x num_frames
e_res = 0.1 # resolution of emissivity array
e_range = np.arange(0.0,1.0,e_res)
num_e = e_range.shape[0]

#I = np.zeros(frames_set_array2.shape,dtype=frames_set_array2.dtype)

P = np.zeros(num_frames,dtype=frames_set_array.dtype)
P2 = np.zeros(num_frames,dtype=frames_set_array.dtype)
P3 = np.zeros(num_frames,dtype=frames_set_array.dtype)
P4 = np.zeros(num_frames,dtype=frames_set_array.dtype)
P5 = np.zeros(num_frames,dtype=frames_set_array.dtype)
w = np.zeros(num_frames,dtype=frames_set_array.dtype)
r0_m2 = np.zeros(num_frames,dtype=frames_set_array.dtype)

Tmax = np.zeros(num_frames,dtype=frames_set_array.dtype)
Tmax2 = np.zeros(num_frames,dtype=frames_set_array.dtype)
##T = Qr2Temp(frames_set_array[:,:,0],e_ss,T0)
##
##I = Temp2IApprox(T,T0,K_spline,Ds_spline,tc)
##
##P = I2Power(I)
##print(P)
##print(estR0Tree(I,pixel_pitch))

T = np.zeros(frames_set_array.shape,dtype=frames_set_array.dtype)
T2 = np.zeros(frames_set_array.shape,dtype=frames_set_array.dtype)
#estimating power and r0 by 

import cv2
cwd_str = os.getcwd()
for f in range(0,num_frames-1,1):
    #print("f=",f)
    if f==0:
        T[:,:,f] = Qr2Temp(frames_set_array[:,:,f],e_ss,T0)
        I = Temp2IApprox(T[:,:,f],T0,Kp_spline,Ds_spline,tc)
    else:
        # prev temp has to be T[:,:,f] to get expected behaviour for Tpeak
        # not sure why
        # if T0 is T[:,:,f-1], then Tpeak rises from 500K to 3200K
        T[:,:,f] = Qr2Temp(frames_set_array[:,:,f],e_ss,T[:,:,f])
        I = Temp2IApprox(T[:,:,f],T[:,:,f-1],Kp_spline,Ds_ispline,tc)

    #print(I)
    # peak temp
    Tmax[f] = np.max(T[:,:,f])
    # power by volume
    #P[f] = I2Power(I)

    # calculate laser power by estimating area using contours, no approx of contours
    thresh,mask,P[f] = estPArea_im2(I,pixel_pitch)
    cv2.imwrite(cwd_str+"\\ThreshImages\\otus-thresh-f{:d}.png".format(f),thresh)
    cv2.imwrite(cwd_str+"\\ContourMask\\no-approx-contour-mask-f{:d}.png".format(f),mask)

    # estimate laser radius using KD-tree
    w[f],P2[f] = estR0Tree(I,pixel_pitch)
    # estimate laser radius by second moment
    r0_m2[f] = estR0_m(I,pixel_pitch)
    #estimate power by KD-tree
    points,P3[f] = estPArea(I,pixel_pitch)
    cv2.imwrite(cwd_str+"\\KDTreePoints\\kdtree-radius-points-f{:d}.png".format(f),points)
    
    # calculate laser power by estimating area using contours, simple chain approx
    # threshold image is not written as it's the same as the one returned from estPArea_im2
    _,mask,P4[f] = estPArea_im(I,pixel_pitch)
    cv2.imwrite(cwd_str+"\\ContourMask\\chain-approx-contour-mask-f{:d}.png".format(f),mask)
    
    # estimate laser power using Canny edge detection to aid in estimating area by contour
    # returns edges mask for saving
    thresh,mask,P5[f] = estPArea_Canny(I,pixel_pitch)
    cv2.imwrite(cwd_str+"\\CannyImages\\canny-edge-f{:d}.png".format(f),thresh)
    cv2.imwrite(cwd_str+"\\ContourMask\\canny-contour-mask-f{:d}.png".format(f),mask)
    
    
f,ax = plt.subplots()
ax.plot(Tmax)
ax.set(xlabel='Frame Idx',ylabel='Max Temp (K)',title='Max Temperature per Frame (K)')
f.savefig('peak-temp.png')

##f,ax = plt.subplots()
##ax.plot(P)
##ax.set(xlabel='Frame Idx',ylabel='Power (W)',title='Power Estimate by Gaussian Volume')
##f.savefig('power-estimate-vol.png')

f,ax = plt.subplots()
ax.plot(P,'b-')
ax.plot(np.full(P.shape,2500.0),'r-')
ax.set(xlabel='Frame Idx',ylabel='Power (W)',title='Power Estimate by Contour, No Approx. (W)')
f.savefig('power-estimate-contour-no-approx.png')

f,ax = plt.subplots()
ax.plot(P2,'b-')
ax.plot(np.full(P2.shape,2500.0),'r-')
ax.set(xlabel='Frame Idx',ylabel='Power (W)',title='Power Estimate by KD-Tree (W)')
ax.set_yscale('log')
f.savefig('power-estimate-kdtree.png')

f,ax = plt.subplots()
ax.plot(P3,'b-')
ax.plot(np.full(P3.shape,2500.0),'r-')
ax.set(xlabel='Frame Idx',ylabel='Power (W)',title='Power Estimate by Area (W)')
f.savefig('power-estimate-area.png')

f,ax = plt.subplots()
ax.plot(P4,'b-')
ax.plot(np.full(P4.shape,2500.0),'r-')
ax.set(xlabel='Frame Idx',ylabel='Power (W)',title='Power Estimate by Contour, Simple Chain (W)')
f.savefig('power-estimate-contour.png')

f,ax = plt.subplots()
ax.plot(P5,'b-')
ax.plot(np.full(P5.shape,2500.0),'r-')
ax.set(xlabel='Frame Idx',ylabel='Power (W)',title='Power Estimate by Contour, Canny (W)')
f.savefig('power-estimate-contour-canny.png')

f,ax = plt.subplots()
ax.plot(w,'b-')
ax.plot(np.full(w.shape,rows*pixel_pitch),'r-')
ax.set_title("Estimated laser radius by tree (m)")
f.savefig('laser-radius-kdtree.png')

f,ax = plt.subplots()
ax.plot(r0_m2,'b-')
ax.plot(np.full(r0_m2.shape,rows*pixel_pitch),'r-')
ax.set(xlabel='Frame Idx',ylabel='Radius (m)',title='Laser Radius Estimate by Second Moment (m)')
f.savefig('laser-radius-m2.png')

plt.show()
print("Finished")
