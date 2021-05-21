from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from skimage.filters import prewitt_v,prewitt_h,prewitt
from skimage.transform import hough_circle, hough_circle_peaks
import os
import cv2
#import cmapy
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

# functions for converting temperature between celcius and kelvin
def CelciusToK(C):
    return float(C)+273.15

def KelvinToC(K):
    return float(K)-273.15

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


def H264ToHDF5(inpath,flag='mask',**kwargs):
    """ Convert a H264 thermal camera video file to HDF5 file

        inpath : Path of the H264 file
        outpath : Path the HDF5 file will be written to
        dsetname : Name of the dataset.
        compress : Method to compress the file.
        copts : Compression options. See H5Py doc.
        flag : Flag passed to readH264. Read doc.
        toC : Convert temperature data to Celcius. Requires toTemp to be set.
        toTemp : Convert the data to temperature Kelvin using Temp2IApprox. Requires e and T0 values
        e : Emissivity.
        T0 : Room temperature in Kelvin

        Reads in the given H264 file using the function readH264 and writes the data array to a HDF5
        file using the set options. This function was designed to be a general purpose converter 

        If the user wants the values to be converted to temperature, the flag
        toTemp must be set and the appropriate emissivity and room temperature values must be provided.
        
        If outpath is None then the file is written to current cwd and
        is based off the H264 file name.

        If dsetname is None, it is set to the name of the input file.

        By default the data is not compressed (False). If the flag is set to True then it is compressed
        using gzip. The user can specify other compression filters as well. The only other compressor
        installed by default with H5py is 'lzf'.

        The compression level for gzip can be set by providing a number from 0-9 to the copts argument.
        Default level used is 4.
    """
    # get filename from input path
    base = os.path.basename(inpath)
    # get filename
    fname = os.path.splitext(base)[0]
    # if output path is set, use it
    # if not base it off the input file name
    if 'outpath' in kwargs:
        outpath = kwargs['outpath']
    elif not 'outpath' in kwargs:
        outpath = fname+ "-HDF5.hdf5"
    # if dsetname is set, use it else set it to filename
    if 'dsetname' in kwargs:
        dsetname = kwargs['dsetname']
    else:
        dsetname = fname
    # if a compression option is given, 
    if 'compress' in kwargs:
        # if it's True, use gzip option
        if kwargs['compress'] == True:
           cpress = 'gzip'
        # if its something else, get the options
        else:
           cpress = kwargs['compress']
    else:
        cpress = None

    if 'copts' in kwargs:
        copts = kwargs['copts']
        if copts<0 or copts > 9:
            raise ValueError('Compression value cannot be negative or greater than 9!')
    else:
        copts = None
        
    # check if convert to temperature flag is set
    if 'toTemp' in kwargs:
        toTemp=kwargs['toTemp']
    else:
        toTemp=False
    # check if convert to Celcius flag is given
    # only set internally if toTemp is given as well
    if 'toC' in kwargs and toTemp:
        toC = kwargs['toC']
    else:
        toC = False
        
    # if to Celcius flag is set, check the required parameters to convert to temperature
    # in the first place are given. If any are missing, raise TypeError
    if toTemp:
        if not 'e' in kwargs:
           raise TypeError('Missing emissivity value!')
        else:
           e = kwargs['e']

        if not 'T0' in kwargs:
           raise TypeError('Missing room temperature value!')
        else:
           T0 = kwargs['T0']
    
    # read in file
    data = readH264(inpath,flag=flag)
    # if convert to temperature flag is set, convert the data to temperature
    if toTemp:
        # convert to Kelvin
        data = Qr2Temp(data,e_ss,T0)
        # convert to Celcius if flag is set
        if toC:
            data -= 273.15
    # create file using context manager to ensure it is closed even in the event of an exception
    with h5py.File(outpath,'w') as f:
        # create dataset using data and compression options
        f.create_dataset(dsetname,data.shape,data=data,compression=cpress,compression_opts=copts)

# path to target H264 file
path = "D:/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"

# starter temperature, room temperature
T0 = CelciusToK(23.0)

## emissivity
# emissivity of stainless steel
e_ss = 0.53

print("Converting to HDF5")
H264ToHDF5(path, # path to h264 values
           toTemp=True,toC=True, # flags to convert values to temperature and to celcius
           e=e_ss,T0=T0, # req values to convert to temperature
           outpath="arrowshape-temperature-HDF5.hdf5", # name of the output file
           compress=True) # flag to compress the data using gzip

