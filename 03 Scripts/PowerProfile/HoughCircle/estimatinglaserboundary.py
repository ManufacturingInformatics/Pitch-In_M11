from scipy import optimize
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.signal import find_peaks
from scipy.fftpack import fftn,ifftn,fftshift
from scipy.spatial.distance import pdist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
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

def FFT2PlotData(data_fft2,hz=True):
    ''' Generates basic plot data from 2D fft data

        fft2_data : 2D fft image
        hz : Flag to convert freq data to Hertz. Default True.

        Returns ordered magnitude and ordered angle data ready for plotting
    '''
    # get flattened magnitude data
    F_mag = np.abs(data_fft2).ravel()
    # get flattened angle data
    F_angle = np.angle(data_fft2).ravel()
    # if hz flag is set, convert angle data to hertz
    if hz:
        F_angle *= 2.0*np.pi
    # sort angle data
    angle_sort = np.argsort(F_angle)
    # return sorted magnitude and freq data so it can be plotted straight away
    return F_mag[angle_sort],F_angle[angle_sort]
        
def filterFreq(I,freq_list,freq_tol=0.5,is_hz=True):
    ''' Filter the given power density matrix for the frequencies listed given the tolerance

        I : Power density matrix
        freq_list : List of frequencies to mask for
        freq_tol : Tolerance on searching for frequencies
        is_hz    : Flag indicating if the frequency list is in hz or rads
        Returns a masked copy of the power density matrix

        Performs a 2D FFT on the given power density matrix and extracts just the
        values that are within range of the given frequencies

        freq_list[i]-freq_tol <= np.angle(fftn(I)) <= freq_list[i]+freq_tol
    '''
    # perform 2D fourier transform of the given power matrix
    I_fft = fftn(I)
    # extract the magnitude and angle
    mag,freq = np.abs(I_fft),np.angle(I_fft)
    # if the target list is in hz, convert freq matrix to hx
    if is_hz:
        freq *= 2.0*np.pi
    # create matrix for boolean search results
    freq_search = np.full(I_fft.shape,False,dtype=np.bool)
    # searches for freqencies that are within the frequency range defined by target freq +/- freq_tol
    for f in freq_list:
        freq_search = np.logical_or(freq_search,np.logical_and(freq>=(f-freq_tol),freq<=(f+freq_tol)))
    # create copy to hold masked values
    I_copy = np.zeros(I_fft.shape,dtype=I_fft.dtype)
    # update values
    I_copy[freq_search]=I_fft[freq_search]
    # return inverse fft of masked results
    return ifftn(I_copy)

def maskFreq(I,freq_list,freq_tol=0.5,is_hz=True):
    ''' Remove the target frequencies from the data I

        I : Power density matrix
        freq_list : List of frequencies to remove
        freq_tol : Tolerance on searching for frequencies
        is_hz    : Flag indicating if the frequency list is in hz or rads
        Returns a masked copy of the power density matrix

        Performs a 2D FFT on the given power density matrix and searches for the values whose
        frequencies are within range.

        freq_list[i]-freq_tol <= np.angle(fftn(I)) <= freq_list[i]+freq_tol

        The indicies that are not found are used to copy the fft values to a new matrix. The
        ifft of that new matrix is returned.

        Returns ifft of matrix whose frequency components are not in the target ranges
        
    '''
    # perform 2D fourier transform of the given power matrix
    I_fft = fftn(I)
    # extract the magnitude and angle
    mag,freq = np.abs(I_fft),np.angle(I_fft)
    # if the target list is in hz, convert freq matrix to hx
    if is_hz:
        freq *= 2.0*np.pi
    # create matrix for boolean search results
    freq_search = np.full(I_fft.shape,False,dtype=np.bool)
    # searches for freqencies that are within the frequency range defined by target freq +/- freq_tol
    for f in freq_list:
        freq_search = np.logical_or(freq_search,np.logical_and(freq>=(f-freq_tol),freq<=(f+freq_tol)))
    # create copy of the fft matrix
    I_copy = np.zeros(I_fft.shape,dtype=I_fft.dtype)
    # update values
    I_copy[~freq_search]=I_fft[~freq_search]
    # return inverse fft of masked results
    return ifftn(I_copy)

def findFilterFreqDiff(Qr,mag_filt=0.05,hz=True,first_inact=0):
    ''' Search the power density matrix for the freq to be used in the filter. Looks for the frequencies
        introduced between an active and inactive frame

        Qr : Data as processed by readH264
        mag_filt : Factor of maximum magnitude to filter the peaks by
        hz : Flag to return frequencies in terms of Hertz. Default True
        first_in : Index of the first frame where the laser is NOT on. Default 0 (i.e. first frame)

        Searches the data to identify the laser spatial frequencies to search for.
        As the peak frequency algorithm picks up lots of little peaks, only the
        frequencies whose magnitudes are above mag_filt times the maximum magnitude
        are returned. This gives helps return only the most dominant components.

        Returns a list of the frequencies of the most prominent components
    '''
    # find the frame with the highest values
    # use it as an example of an active frame where the laser is turned on
    d_max = np.unravel_index(np.argmax(Qr),Qr.shape)[2]
    # find the frequencies introduced between first inactive frame and the found active frame
    I_diff = fftn(Qr[:,:,d_max])-fftn(Qr[:,:,first_inact])
    # get the magnitude and frequency data for the fft difference
    mm,ff = FFT2PlotData(I_diff,hz=hz)
    # search for peaks, reutrns the indicies
    mag_peaks = find_peaks(mm)[0]
    # filter peaks for just the most prominent components
    mag_peaks = mag_peaks[mm[mag_peaks]>(mag_filt*np.max(mm[mag_peaks]))]
    # return the frequencies associated with these peaks
    return ff[mag_peaks]

def findFilterFreq(M,mag_filt=0.05,hz=True,return_index=True):
    ''' Search the power density matrix for the freq to be used in the filter. Looks for the frequencies
        introduced between an active and inactive frame

        M : Matrix to search
        mag_filt : Factor of maximum magnitude to filter the peaks by
        hz : Flag to return frequencies in terms of Hertz. Default True
        first_in : Index of the first frame where the laser is NOT on. Default 0 (i.e. first frame)

        Searches the data to identify the laser spatial frequencies to search for.
        As the peak frequency algorithm picks up lots of little peaks, only the
        frequencies whose magnitudes are above mag_filt times the maximum magnitude
        are returned. This gives helps return only the most dominant components.

        Returns a list of the frequencies of the most prominent components
    '''
    # find the frequencies introduced between first inactive frame and the found active frame
    I_diff = fftn(M)
    # get the magnitude and frequency data for the fft difference
    mm,ff = FFT2PlotData(I_diff,hz=hz)
    # search for peaks, reutrns the indicies
    mag_peaks = find_peaks(mm)[0]
    # filter peaks for just the most prominent components
    mag_peaks = mag_peaks[mm[mag_peaks]>(mag_filt*np.max(mm[mag_peaks]))]
    # return the frequencies associated with these peaks
    if not return_index:
        return ff[mag_peaks]
    else:
        return ff[mag_peaks],mag_peaks

def findPeakFilterPerc(peaks_list,hist_bins=7):
   ''' Calculate by what percentage of the maximum the peaks should be filtered by

       peaks_list : Collection of unfiltered peak values
       hist_bins : Number of bins to use in histogram

       This function returns the ideal percentage of the maximum the peaks can be filtered by to get only
       the most dominant peaks

             peaks_list[peaks_list >= max(peaks_list)*perc]

       A histogram is used to identify which percentage bin the majority of values belongs to. The maximum value
       in that bin is used to calculate the returned percentage value.

       Returns percentage value to be used in boolean indexing
   '''
   # calculate the histogram of the values based on percentage of maximum
   # returns populations and edges of those bins
   max_peak = np.max(peaks_list)
   n,edges = np.histogram(peaks_list/max_peak,bins=hist_bins)
   # find which bin has the highest population
   # sorted ascending
   nn = n.argsort()
   # Find which values are in the target bin
   peaks_bin = np.where(np.logical_and(peaks_list/max_peak>=edges[nn[-1]],peaks_list/max_peak<=edges[nn[-2]]))
   # Calculate the percentage as the maximum value in the bin divided by the max result
   # the list is treated as an array to allow slicing and save space
   return np.asarray(peaks_list)[peaks_bin].max()/max_peak

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

def drawHoughCircles(circles,shape,**kwargs):
    ''' Draw the givenn list of circles from the HoughCircles function onto a blank matrix

        circles : Array of circles data returned by OpenCVs HoughCircles function
        shape : Shape of the mask to draw results on. Mask size is (shape,3)
        **kwargs : Additional arguments to customise drawing
            centre_col : color to draw circle centres with, tuple of uint8 values
            bounday_col : color to draw circle boundaries with, tuple of uint8 values
            iter_cols : Each circle is drawn with a unique color and centre color decided by it's order in the matrix.
                        (255/ci,255/ci,255/ci). Centre and boundary color ARE THE SAME

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
        
    # blank matrix
    mask = np.zeros((*shape,3),dtype='uint8')
    # iterate through circles array
    # array is [1xCx3]
    len_c = circles.shape[1]
    for c in range(len_c):
        # draw circle boundary
        cv2.circle(mask, (circles[0,c,0], circles[0,c,1]), circles[0,c,2], bcol, 1)
    for c in range(len_c):
        cv2.circle(mask, (circles[0,c,0], circles[0,c,1]), 1, ccol, 1)
    # return circle result
    return mask

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

# path to footage
path = "C:/Users/uos/Documents/BEAM/Scripts/LMAP Thermal Data/Example Data - _Tree_/video.h264"
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

print("Writing difference plots to file")
diff_folder = os.getcwd()+'\DiffMatricies\\'
# create axes
fig,ax = plt.subplots()
# create colorbar axes
cax = make_axes_locatable(ax).append_axes("right",size="5%",pad="2%")
for f in range(I.shape[2]):
   # calculate sum difference plot
   diff_all = diffRow(I[:,:,f]) + diffCol(I[:,:,f])
   # create filled contour plot
   diff_ct = ax.contourf(diff_all,cmap='hsv')
   # clear the colorbar axes
   cax.clear()
   # create new colorbar
   fig.colorbar(diff_ct,cax=cax)
   # update axes labels
   ax.set_title('Difference Plot of Frame {0}'.format(f))
   # save fig
   fig.savefig(diff_folder+'diff-sum-f{0}.png'.format(f))
   
# laser radius
r0 = 0.35e-3
w0 = 2*r0
# laser radius interms of pixels
r0_p = int(np.ceil(r0/pixel_pitch))
# waist size assuming gaussian distribution
# 4 times std 
rmax_gauss = 4*r0_p

## Using frame 1048 as a starter point
# difference row wise
diff_row = diffRow(I[:,:,1048])
# difference col wise
diff_col = diffCol(I[:,:,1048])
# sum of the two
diff_all = diff_row + diff_col
# perform HoughCircles
print("Computing Hough Circles for r0_p and rmax_gauss")
circles = cv2.HoughCircles(diff_all.astype('uint8'),
                           cv2.HOUGH_GRADIENT, # method, only one supported
                           1,       # inv. ratio of accumulator resoluton to image resolution
                           rows/8,  # min. distacne between centres of circles
                           param1=100, # higher threshold of the two passed to the Canny edge detector
                           param2=30,  # accumulator threshold for the circle centres at the detection stage
                           minRadius=r0_p, # minm radius
                           maxRadius=rmax_gauss) # maxm radius

# draw results
circles_res = drawHoughCircles(circles,diff_all.shape)
cv2.imwrite('houghcircle-results-rmin-rmax.png',circles_res)
## radii stats
# extract just the radii stats
frame_radii = [c[2] for c in circles[0,:]]
print("Radii range: ",(max(frame_radii)-min(frame_radii))*pixel_pitch,"m")
# perform histogram
pop,edges = np.histogram(frame_radii,bins=7)
# create plot
f,ax = plt.subplots()
# create bar plot
ax.bar((edges[:-1]+edges[1:])/2,pop,width=(edges[1]-edges[0]),align='center')
ax.set(xlabel='Hough Circle Radii (pixels)',ylabel='Bin Population',title='Histogram of Radii of the Circles found by Hough Circle using r0 and 4*r0')
# save bar plot
f.savefig('houghcircles-rmin-rmax-hist-f1048.png')

## Trying hough circles using max radii with a tolerance
# apply two bin histogram to the circles found in previous run
# most values are background values
twob_pop,twob_edges = np.histogram(frame_radii,bins=2)
# sort populations in descending order
twob_pop_sort = twob_pop.argsort()
# find the bin edges or radii limits for the least occupied bin
min_rad = twob_edges[twob_pop_sort[0]]
max_rad = twob_edges[twob_pop_sort[0]+1]
# draw just those circles
print("Calculating Hough Circles using limits discovered by two bin histogram")
circles = cv2.HoughCircles(diff_all.astype('uint8'),
                           cv2.HOUGH_GRADIENT, # method, only one supported
                           1,       # inv. ratio of accumulator resoluton to image resolution
                           rows/8,  # min. distacne between centres of circles
                           param1=100, # higher threshold of the two passed to the Canny edge detector
                           param2=30,  # accumulator threshold for the circle centres at the detection stage
                           minRadius=int(np.ceil(min_rad-5)), # minm radius
                           maxRadius=int(np.ceil(max_rad+5))) # maxm radius

# draw results
circles_res = drawHoughCircles(circles,diff_all.shape)
cv2.imwrite('houghcircle-results-rmax-tol.png',circles_res)
## radii statsextract just the radii stats
frame_radii = [c[2] for c in circles[0,:]]
print("Radii range: ",(max_rad-min_rad)*pixel_pitch,"m")
# perform histogram
pop,edges = np.histogram(frame_radii,bins=7)
# create bar plot
f,ax = plt.subplots()
ax.bar((edges[:-1]+edges[1:])/2,pop,width=(edges[1]-edges[0]),align='center')
ax.set(xlabel='Hough Circle Radii (pixels)',ylabel='Bin Population',title='Histogram of Radii of the Circles found by Hough Circle using Least Populated Bin')
# save bar plot
f.savefig('houghcircles-rmax-tol-hist-f1048.png')

## Trying a masked portion of the matrix
# because of the offset between the camera and the laser, the laser only appears in approx left 2/3 of the image
diff_all_part = np.zeros((rows,cols),dtype='uint8')
diff_all_part[:,:]=diff_all.astype('uint8')[:,:]
# set right third of image to 0
diff_all_part[:,int(np.ceil(cols*(2/3))):] = 0
print("Calculating Hough Circles using masked image")
circles = cv2.HoughCircles(diff_all_part,
                           cv2.HOUGH_GRADIENT, # method, only one supported
                           1,       # inv. ratio of accumulator resoluton to image resolution
                           rows/8,  # min. distacne between centres of circles
                           param1=100, # higher threshold of the two passed to the Canny edge detector
                           param2=30,  # accumulator threshold for the circle centres at the detection stage
                           minRadius=0, # minm radius
                           maxRadius=rmax_gauss) # maxm radius

# draw results
circles_res = drawHoughCircles(circles,diff_all.shape)
cv2.imwrite('houghcircle-results-diff-part-rmin-rmax.png',circles_res)
f,ax = plt.subplots(1,2)
ax[0].contourf(diff_all,cmap='hsv')
# draw filled contour of the circle results
# image converted to gray so contour can generate colors
ax[1].contourf(cv2.cvtColor(circles_res,cv2.COLOR_BGR2GRAY))
f.suptitle('Circles found in the Left Two Thirds of the Difference Matrix')
ax[0].set_title('Difference Matrix')
ax[1].set_title('Circles Found')
f.savefig('houghcircles-diff-part-results-compare.png')

## radii stats
# extract just the radii stats
frame_radii = [c[2] for c in circles[0,:]]
print("Radii range: ",(max(frame_radii)-min(frame_radii))*pixel_pitch,"m")
# perform histogram
pop,edges = np.histogram(frame_radii,bins=7)
# create plot
f,ax = plt.subplots()
# create bar plot
ax.bar((edges[:-1]+edges[1:])/2,pop,width=(edges[1]-edges[0]),align='center')
ax.set(xlabel='Hough Circle Radii (pixels)',ylabel='Bin Population',title='Histogram of Radii of the Circles found in Left 2/3 of Diff using r0 and 4*r0')
# save bar plot
f.savefig('houghcircles-diff-part-rmin-rmax-hist-f1048.png')

## Taking the largest circle
# sort by radius
cc_idx = np.argsort([c[2] for c in circles[0]])[::-1]
P_diff = 1000
best_ci = 0
# iterate through circles to find the power estimates
print("Calculating power for each circle found")
for ci in cc_idx:
    mask = np.zeros((rows,cols),dtype='uint8')
    # draw largest circle on the mask filled
    cv2.circle(mask,(*circles[0,ci][:2],),circles[0,ci][2],(255),cv2.FILLED)
    i,j = np.where(mask==255)
    # sum the power density values in that area and multiply by area
    P_ci = np.sum(I[i,j,1048])*(np.pi*(circles[0,ci][2]*pixel_pitch)**2.0)
    print("C:{0},P:{1}W,r:{2}m".format(ci,P_ci,circles[0,ci][2]*pixel_pitch))
    # update closest circle power estimate
    if np.abs(P_ci-500.0)<P_diff:
       P_diff = np.abs(P_ci-500.0)
       best_ci = ci

print("Best circle is {0} with a power estimate of {1} W".format(best_ci,powerEstHoughCircle(circles[:,best_ci,:][0],I[:,:,1048],pixel_pitch)))
print("Corresponding radius {0} m".format(circles[:,best_ci,2][0]*pixel_pitch))
# radius of the circe that gives the best radius
best_r = circles[:,best_ci,2][0]
# convert best radius to integer so it can be used
best_r_c = int(np.ceil(best_r))
      
## Using average distance and radius
avg_radii = int(np.mean(frame_radii))
avg_centre_y = int(np.mean([c[1] for c in circles[0,:]]))
avg_centre_x = int(np.mean([c[0] for c in circles[0,:]]))
mask = np.zeros((rows,cols),dtype='uint8')
cv2.circle(mask,(avg_centre_x,avg_centre_y),avg_radii,(255),cv2.FILLED)
i,j = np.where(mask==255)
print("Power using Average Circle {0} W".format(np.sum(I[i,j,1048])*(np.pi*(avg_radii*pixel_pitch)**2.0)))

## Filtering using histogram
print("Searching using the middle bin values")
# create copy of difference matrix
diff_all_filt = np.zeros((rows,cols),dtype='uint8')
diff_all_filt[:,:]=diff_all_part[:,:]
# perform three bin histogram
# bk values, boundary values and peak values
pop,edges = np.histogram(diff_all_filt,bins=3)
# sort by population, ascending order
pp = pop.argsort()
# as we have 3 bins, when sorted the boundary values are the middle bin
# get the limits
middle_bin = edges[pp[1]:pp[1]+2]
# set the values outside of the middle bin to 0
diff_all_filt[np.logical_and(diff_all_filt<middle_bin[0],diff_all_filt>middle_bin[1])]=0.0
# perform hough circle transform on the result
circles = cv2.HoughCircles(diff_all_part,
                           cv2.HOUGH_GRADIENT, # method, only one supported
                           1,       # inv. ratio of accumulator resoluton to image resolution
                           rows/8,  # min. distacne between centres of circles
                           param1=100, # higher threshold of the two passed to the Canny edge detector
                           param2=30,  # accumulator threshold for the circle centres at the detection stage
                           minRadius=0, # minm radius
                           maxRadius=rmax_gauss) # maxm radius

# draw results
circles_res = drawHoughCircles(circles,diff_all.shape)
cv2.imwrite('houghcircle-results-diff-part-filt-rmin-rmax.png',circles_res)
f,ax = plt.subplots(1,2)
ax[0].contourf(diff_all,cmap='hsv')
# draw filled contour of the circle results
# image converted to gray so contour can generate colors
ax[1].contourf(cv2.cvtColor(circles_res,cv2.COLOR_BGR2GRAY))
f.suptitle('Circles found in the Middle Histogram Values of the Masked Difference Matrix')
ax[0].set_title('Difference Matrix')
ax[1].set_title('Circles Found')
f.savefig('houghcircles-diff-part-filt-results-compare.png')

## Impact of different masking percentages on the closest power estimate
print("Trying different masking percentages")
diff_all_part = np.zeros((rows,cols),dtype='uint8')
# set right third of image to 0
mask_perc = np.linspace(0.0,1.0,2000)
closest_Pest = np.zeros(mask_perc.shape)
for pi,perc in enumerate(mask_perc):
   #reset difference matrix
   diff_all_part[:,:]=diff_all.astype('uint8')[:,:]
   # clear set percentage of values in the matrix
   diff_all_part[:,int(np.ceil(cols*perc)):] = 0
   # find circles
   circles = cv2.HoughCircles(diff_all_part,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
   # if circles were found
   if circles is not None:
      # evaluate the power estimate for each circle and find the smallest distance to 500.0
      # update matrix with that value
      closest_Pest[pi] = np.abs(500.0-np.apply_along_axis(powerEstHoughCircle,2,circles,I=I[:,:,1048],PP=pixel_pitch)).min()

## plot results
f,ax = plt.subplots(figsize=[s*1.5 for s in plt.rcParams['figure.figsize']])
# plot values
ax.plot(mask_perc*100.0,closest_Pest)
# add labels to the matrix
ax.set(xlabel='Masking Percentage (%)',ylabel='Difference to Closest Power Estimate (W)',title='Impact of Masking Different Percentages of the Difference Matrix on the Closest Power Estimate')
f.savefig('mask-perc-closest-pest.png')

## Trying the entire dataset
print("Applying Hough Transform to the entire dataset and storing metrics")
# folder for hough circle results
hough_folder = os.getcwd()+'\HoughCircles\\'
# best power estimate relative to 500W
best_pest = np.zeros(I.shape[2])
# index of best circle
best_ci = np.zeros(I.shape[2])
# radius of the best circle
best_r = np.zeros(I.shape[2])
# range of power estimate
best_prange = np.zeros(I.shape[2])
# average distance between centres
avg_dist_c = np.zeros(I.shape[2])
# number of circles found
circles_found = np.zeros(I.shape[2])

for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
   ## Evaluate results
   if circles is not None:
        # write found circles to file
        #cv2.imwrite(hough_folder+'houghcircle-results-f{0}.png'.format(ff),drawHoughCircles(circles,diff_all.shape))
        # estimate the power estimate for each circle
        pest = np.apply_along_axis(powerEstHoughCircle,2,circles,I=I[:,:,ff],PP=pixel_pitch)[0]
        # sort power estimates by how close it is to the 500W line
        pest_asort = np.abs(500.0-pest).argsort()
        # update best power estimate values
        best_pest[ff] = pest[pest_asort[0]]
        # update index of best circle
        best_ci[ff] = pest_asort[0]
        # update best circle radius
        best_r[ff] = circles[:,pest_asort[0],2]*pixel_pitch
        # update power estimate range
        best_prange[ff] = np.abs(pest.max()-pest.min())
        # average distance between centres
        avg_dist_c[ff] = np.mean(pdist(circles[0,:,:2]))
        # number of circles found
        circles_found[ff] = circles.shape[1]

# get rid of nans
np.nan_to_num(best_pest,copy=False)
np.nan_to_num(best_ci,copy=False)
np.nan_to_num(best_r,copy=False)
np.nan_to_num(best_prange,copy=False)
np.nan_to_num(avg_dist_c,copy=False)
np.nan_to_num(circles_found,copy=False)

# create axis, they will be reused
# size scale factor
fig_scale = 2.0
fig_size = [s*fig_scale for s in plt.rcParams['figure.figsize']]
f,ax = plt.subplots(figsize=fig_size)
# best power estimate
ax.plot(best_pest)
f.suptitle('Best Power Estimate Found Using Hough Circle Transform')
ax.set(xlabel='Frame Index',ylabel='Best Power Estimate (W)')
f.savefig('best-power-estimate-houghcircle.png')
# index of best circle
ax.clear()
ax.plot(best_ci)
f.suptitle('Index of the Circle that Gives the Closest Power Estimate')
ax.set(xlabel='Frame Index',ylabel='Index of Best Circle')
f.savefig('best-circle-index-houghcircle.png')
# best circle radius
ax.clear()
ax.plot(best_r)
f.suptitle('Radius of the Circle that Gives the Closest Power Estimate')
ax.set(xlabel='Frame Index',ylabel='Radius of the Best Circle (m)')
f.savefig('best-circle-radius-houghcircle.png')
# Power Range
ax.clear()
ax.plot(best_prange)
f.suptitle('Range of Power Estimates Found using Hough Circle Transform')
ax.set(xlabel='Frame Index',ylabel='Range of Power Estimates (W)')
f.savefig('pest-range-houghcircle.png')
# Average Distance between Centres
ax.clear()
ax.plot(avg_dist_c)
f.suptitle('Average Distance Between Centres of Found Hough Circles')
ax.set(xlabel='Frame Index',ylabel='Avg. Distance between Centres (Pixels)')
f.savefig('avg-dist-circle-centres-houghcircle.png')
# Number of circles found
ax.clear()
ax.plot(circles_found)
f.suptitle('Number of Circles Found in Power Density Frames')
ax.set(xlabel='Frame Index',ylabel='Number of Circles Found')
f.savefig('num-circles-houghcircle.png')

print("Applying results from metrics about dataset results")
### Trying different limits to see impact on power plot
## Choosing the first circle
pest = np.zeros(I.shape[2])
print("Trying first circle")
for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
   if circles is not None:
      # calculate power using the first circle
      pest[ff] = powerEstHoughCircle(circles[:,0,:][0],I=I[:,:,ff],PP=pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate using only first found Circle')
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-first-circle-houghcircles.png')

## Trying average circle
pest = np.zeros(I.shape[2])
ci_mean = int(np.floor(best_ci.mean()))
print("Trying average circle, {0}".format(ci_mean))
for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
   if circles is not None:
      # if there are not enough circles to use ci_mean th circle, use last circle
      pest[ff] = powerEstHoughCircle(circles[:,min(circles.shape[1]-1,ci_mean),:][0],I=I[:,:,ff],PP=pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate using {0}th Circle'.format(ci_mean))
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-mean-th-circle-houghcircles.png')

## Changing min distance between circles
pest = np.zeros(I.shape[2])
print("Trying min distance between circles. Mean value")
ccdist = int(np.floor(avg_dist_c.mean()))
for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,ccdist,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
   if circles is not None:
      # calculate power using the first circle
      pest[ff] = powerEstHoughCircle(circles[:,0,:][0],I[:,:,ff],pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate using a Min. Distance Between Centres of {0} P'.format(ccdist))
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-centre-dist-houghcircles.png')

## Changing min radius
pest = np.zeros(I.shape[2])
# use mean of best radii
rmin = int(np.floor(best_r.mean()/pixel_pitch))
print("Trying avg. best radius")
for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=rmin-10,maxRadius=rmin+15)
   if circles is not None:
      # calculate power using the first circle
      pest[ff] = powerEstHoughCircle(circles[:,0,:][0],I[:,:,ff],pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate using first Circle and Mean Best Radius of {0} pixels'.format(rmin))
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-mean-best-r-houghcircles.png')

## Changing max radius
pest = np.zeros(I.shape[2])
# previous run identified the best radius as 42.7 pixels
# ratio to r0 is 2.44
rmin=int(np.floor(r0/pixel_pitch))
rmax=int(np.ceil((r0*2.44)/pixel_pitch))
print("Trying radius range as r0 to 2.44*r0")
for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=rmin,maxRadius=rmax+5)
   if circles is not None:
      # calculate power using the first circle
      pest[ff] = powerEstHoughCircle(circles[:,0,:][0],I[:,:,ff],pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate using Radii Range of r0 to 2.44*r0')
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-custom-r-range-houghcircles.png')

## Changing max radius number 2
pest = np.zeros(I.shape[2])
# identifying the ratio between r0 and the the radius that was closest to r0
rbest = best_r[np.abs(500.0-best_pest).argmin()]
rmax=int(np.ceil(rbest/pixel_pitch))
print("Trying radius range of [{0},{1}], based off radius of closest estimate".format(rmax-10,rmax))
for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=rmax-10,maxRadius=rmax)
   if circles is not None:
      # calculate power using the first circle
      pest[ff] = powerEstHoughCircle(circles[:,0,:][0],I[:,:,ff],pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate using Radii Range of r0 to Radius of Closest Circle,r={:.5f}m'.format(rbest))
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-best-r-range-houghcircles.png')

## Changing max radius AND using largest circle
pest = np.zeros(I.shape[2])
# previous run identified the best radius as 42.7 pixels
# ratio to r0 is 2.44
rmin=int(np.floor(r0/pixel_pitch))
rmax=int(np.ceil((r0*2.44)/pixel_pitch))
print("Trying radius range as r0 to 2.44*r0, with largest circle")
for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=rmin,maxRadius=rmax)
   if circles is not None:
      # calculate power using the largest circle (by radius)
      pest[ff] = powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate using Radii Range of r0 to 2.44*r0 and Largest Circle')
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-custom-r-range-biggest-circle-houghcircles.png')

## Searching for radius range that satisfies the known laser spec
pest = np.zeros(I.shape[2])
# tolderance on laser power
power_tol = 0.03
# values where the power values are within tolerance
idx = np.where(np.abs(500.0-best_pest)<(500.0*power_tol))[0]
# radius range
rmin = int(np.floor(best_r[idx].min()/pixel_pitch))
rmax = int(np.ceil(best_r[idx].max()/pixel_pitch))
# plot marking the radii that are within range
ax.clear()
ax.plot(best_r,'b-',idx,best_r[idx],'rx')
f.suptitle('Radius of the Circle that Gives the Closest Power Estimate')
f.legend(['All Values','Spec.'])
ax.set(xlabel='Frame Index',ylabel='Circle Radius (m)')
f.savefig('circle-r-power-spec-houghcircles.png')

for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=rmin,maxRadius=rmax+5)
   if circles is not None:
      # calculate power using the first circle
      pest[ff] = powerEstHoughCircle(circles[:,0,:][0],I[:,:,ff],pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate using Radii Range Based off Power Estimates within Specification')
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-power-spec-r-range-houghcircles.png')

## Searching a wide range but only evaluating circles within a specified range
print("Trying a wide range with a targeted search range")
pest = np.zeros(I.shape[2])
r_range = [int(np.floor(0.0002/pixel_pitch)),int(np.ceil(0.00038/pixel_pitch))]
for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
   if circles is not None:
      # get the circles within the target range
      cc =circles[0,np.logical_and(circles[0,:,2]>=r_range[0],circles[0,:,2]<=r_range[1]),:]
      # if there are circles in the target range
      if cc.shape[0]>0:
         # choose the largest one
         # sort by radii in ascending order and choose the last element, the largest one
         pest[ff] = powerEstHoughCircle(cc[cc[:,2].argsort()[-1],:],I[:,:,ff],pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate using the largest Radii within a Target Range [{0},{1}] P'.format(*r_range))
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-search-target-range.png')

## Removing out of bounds circles from the arrays
print("Trying largest circle while removing out of bounds circles")
pest = np.zeros(I.shape[2])
for ff in range(I.shape[2]):
   # get difference matrix
   diff_all = (diffRow(I[:,:,ff])+diffCol(I[:,:,ff])).astype('uint8')
   # mask right third of the image
   diff_all[:,int(np.ceil(cols*(2/3))):] = 0
   # perform hough circle algorithm
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
   if circles is not None:
      # removing out of bounds circles
      circles = remCirclesOutBounds(circles,I.shape[:2])
      # if there are circles left
      if circles.shape[1]>0:
         pest[ff] = powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

# plot results
ax.clear()
ax.plot(pest)
f.suptitle('Power Estimate when Filtering Out of Bounds Circles')
ax.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
f.savefig('pest-filter-outofbounds.png')
