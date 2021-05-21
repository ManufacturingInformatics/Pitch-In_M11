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
   ''' Remove the circles from the set of circles whose estimated circle area goes out of the image

       circles : Array containing [centre_x,centre_y,radius] for each circle found
       shape : Shape of the data they were found in

       This function checks if given the postion and radius of the circle whether any portion of it
       goes outside the bounds of shape.

       Returns filtered list of circles
   '''
   # search for any circles that are out of bounds
   outofbounds_idx = np.where(np.logical_or(
      np.logical_or(circles[0,:,0]-circles[0,:,2]<0,circles[0,:,0]+circles[0,:,2]>shape[1]), # checking if any x coordinates are out of bounds
      np.logical_or(circles[0,:,1]-circles[0,:,2]<0,circles[0,:,1]+circles[0,:,2]>shape[1])))[0] # checking if any y coordinates are out of bounds
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

def remCirclesROI(circles,xlim,ylim):
   ''' Remove the circles from the target region of interest in the image

       circles: Array containing [centre_x,centre_y,radius] for each circle found
       xlim : Two element array describing the lower and upper x limits for the region of intererst [lower,upper]
       ylim : Two element array describing the lower and upper y limits for the region of interest [lower,upper]

       This function checks to see if any portion of the specified circles goes outside the limits given. If it does,
       it is removed from the array.

       If xlim or ylim are single element arrays then it is assumed to be the upper limits. e.g. ylim = 128 => [0,128]
       If any of the elements are negative, then -1 is returned.

       Returned filtered list of circles whose areas are within the limits specified by the user
   '''
   # if the limits are only one element in length
   if len(xlim)==1:
      xup = xlim
      xdwn = 0
   else:
      xup = xlim[1]
      xdwn = xlim[0]

   if len(ylim)==1:
      yup = ylim
      ydwn = 0
   else:
      yup = ylim[1]
      ydwn = ylim[0]

   # if any of the limits passed are negative, return -1
   if (np.asarray(xlim)<0).any():
      return -1

   if (np.asarray(ylim)<0).any():
      return -1
         
   # search for any circles that are out of bounds
   outofbounds_idx = np.where(np.logical_or(
      np.logical_or(circles[0,:,0]-circles[0,:,2]<xdwn,circles[0,:,0]+circles[0,:,2]>xup), # checking if any x coordinates are out of bounds
      np.logical_or(circles[0,:,1]-circles[0,:,2]<ydwn,circles[0,:,1]+circles[0,:,2]>yup)))[0] # checking if any y coordinates are out of bounds
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

def maskCirclesRadius(circles,rmin,rmax):
   ''' Remove circles whose radii are outside the limits

       circles : Array containing [centre_x,centre_y,radius] for each circle found
       rmin : Min. radius in terms of pixels
       rmax : Max. radius in terms of pixels

       Searches for circles whose radii are too small or too big as per the limits set by
       arguments rmin and rmax. The returns list has circles are within the limits.

       rmin <= r <= rmax

       Returns the filtered list of circles
   '''
   # searches for circles that are too small or too big
   outofbounds_idx = np.where(np.logical_or(circles[0,:,2]<rmin,circles[0,:,2]>rmax))[0]
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
   
def customOpenAction(src,kernel,**kwargs):
   ''' Custom opening operation applied to source image

      src : Source 8-bit image
      kernel : Kernel applied to the source image during BOTH the erode and dilate operations
      **kwargs : Customization parameters
         erode_it : number of times the eroding operation is applied
         dilate_it : number of times the dilate operation is applied
         erode_k : Kernel for erode operation, Overrides the kernel arg
         dilate_k : Kernal for dilate operation. Overrides the kernel arg


      Applies a customized open morphological transform to the source image. The listed supported arguments for the kwargs
      kwargs show how it can be customised. If no additional options are given then, the kernel is applied once in the open
      operation and then once in the closed operation
      Returns the image after the specified open operation
   '''
   # check kwargs
   if 'erode_it' in kwargs.keys():
      erode_it = kwargs['erode_it']
   else:
      erode_it = 1

   if 'dilate_it' in kwargs.keys():
      dilate_it = kwargs['dilate_it']
   else:
      dilate_it = 1

   if 'erode_k' in kwargs.keys():
      erode_k = kwargs['erode_k']
   else:
      erode_k = kernel

   if 'dilate_k' in kwargs.keys():
      dilate_k = kwargs['dilate_k']
   else:
      dilate_k = kernel

   # apply erode operation
   ero = cv2.erode(src,erode_k,iterations=erode_it)
   # apply dialate operation
   dia = cv2.dilate(ero,dilate_k,iterations=dilate_it)

   return dia

def findLaserBorderBest(I,PP,r0,rr=3.5):
   ''' Find the ideal laser border given the investigation into the best parameters

       I : image to search
       PP : pixel pitch
       r0 : laser radius
       rr : Ratio between laser radius and the radius of the laser boundary. Dependent on height
   '''
   if I.dtype != np.uint8:
      I_img = I.astype('uint8')
   else:
      I_img = I
   # find the difference matrix
   diff_all = diffRow(I_img)+diffCol(I_img)
   # mask the right third of the image
   diff_all[:,int(I.shape[0]*(2/3))]=0
   # perform open operation
   res = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
   # search for circles across a wide range
   circles = cv2.HoughCircles(res,cv2.HOUGH_GRADIENT,1,int(I.shape[0]/8),param1=100,param2=30,minRadius=0,maxRadius=4*r0)
   # if circles were found
   if circles is not None:
      # remove out of bounds circles
      circles = remCirclesOutBounds(circles,I.shape[:2])
      # if there are still circles left
      if circles.shape[1]>0:
         # calculate power using that circle
         return powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I,PP)
      else:
      # if there are no circles after filtering
         return 0.0
   # if there are no circles, return 0.0
   else:
      return 0.0
         
def estimateCircleEntropy(circle,I):
   ''' Estimate the Shannon Entropy within the circle

       circle : [x,y,r]
       I : power density matrix the circles were detected in

       Finds the pixels within the area of the circle and calculate the entropy of this area
       using the Shannon Entropy approach. The values within the circle are copied over to a square
       where the values outside of the circle are 0

       Returns the entropy within the circle roi
   '''
   from skimage.filter.rank import entropy
   # find the shape of the target image
   # use it to create a meshgrid of coordinates
   xx,yy = np.meshgrid(np.arange(0,I.shape[0]),np.arange(0,I.shape[1]))
   # find which pixels are within range
   xr,yr = np.where((((xx-circle[0])**2.0)+((yy-circles[1])**2.0))**0.5 <= circles[2])
   # create blank matrix for results
   vals = np.zeros(I.shape[:2],np.uint8)
   # copy values over
   vals[xr,yr] = I[xr,yr]
   # calculate entropy using the values
   return entropy(vals)
   
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

# laser radius
r0 = 0.00035 #m
# laser radius in pixels
r0_p = int(np.ceil(r0/pixel_pitch))
# assuming gaussian behaviour, 99.7% of values are within 4 std devs of mean
# used as an upper limit in hough circles
rmax_gauss = 4*r0_p

# range of kernel sizes to try
k_size = list(range(2,6))

# different folders for the results
open_results = os.getcwd() + '/OpenResults//'
close_results = os.getcwd() + '/CloseResults//'
close_open_results = os.getcwd() + '/OpenCloseResults//'
open_results_I = os.getcwd() + '/OpenResults-I//'
close_results_I = os.getcwd() + '/CloseResults-I//'
shape_results = os.getcwd() + '/OpenResults-KernelShapes//'

####################################################################################################################################
## Impact of filtering the out of bounds circles on the result
# RAW values
print("Calculating power using largest circle and filtering OOB circles,raw difference matrix")
pest = np.zeros(I.shape[2])
pest_filt = np.zeros(I.shape[2])
f,ax = plt.subplots(1,2)
for ff in range(I.shape[2]):
    diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
    # search for circles in the image
    circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        pest[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)
        # filter the circles
        circles = remCirclesOutBounds(circles,(rows,cols))
        # if there are circles left
        if circles.shape[1]>0:
           pest_filt[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)
           
ax[0].plot(pest)
ax[1].plot(pest_filt)
ax[0].set(xlabel='Frame Index',ylabel='Est. Power (W)',title='Original')
ax[1].set(xlabel='Frame Index',ylabel='Est. Power (W)',title='Filtered')
f.suptitle('Est. Power of the Largest Circle Before and After Filtering Circles,Raw')
f.savefig('pest-compare-filt-diff-matrix.png')

# OPENING values
print("Calculating power using largest circle and filtering OOB circles,opened difference matrix")
pest = np.zeros(I.shape[2])
pest_filt = np.zeros(I.shape[2])
for ff in range(I.shape[2]):
    diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
    # erode then dilate
    open_diff = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # save opening preprocessing results
    cv2.imwrite(open_results + 'open-results-f{0}.png'.format(ff),open_diff)
    # search for circles in the image
    circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        pest[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)
        # filter the circles
        circles = remCirclesOutBounds(circles,(rows,cols))
        # if there are circles left
        if circles.shape[1]>0:
           pest_filt[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)
# clear axes
ax[0].clear()
ax[1].clear()
# plot power estimates
ax[0].plot(pest)
ax[1].plot(pest_filt)
ax[0].set(xlabel='Frame Index',ylabel='Est. Power (W)',title='Original')
ax[1].set(xlabel='Frame Index',ylabel='Est. Power (W)',title='Filtered')
f.suptitle('Est. Power of the Largest Circle Before and After Filtering Circles,Open')
f.savefig('pest-compare-filt-open-diff-matrix.png')

####################################################################################################################################
## Trying opening with different kernel sizes and saving the power plot
# still using the largest circle
f,ax = plt.subplots()
for s in k_size:
    print("Trying opening, size ",s)
    pest_open = np.zeros(I.shape[2])
    for ff in range(I.shape[2]):
        diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
        # erode then dilate
        open_diff = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,np.ones((s,s),np.uint8))
        # save opening preprocessing results
        cv2.imwrite(open_results + 'open-results-f{0}-s{1}-{1}.png'.format(ff,s),open_diff)
        # search for circles in the image
        circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
        # if circles were found
        if circles is not None:
            # calculate power using the largest circle
            pest_open[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)
            
    ax.clear()
    ax.plot(pest_open)
    ax.set(xlabel='Frame Index',ylabel='Est. Power (W)')
    f.suptitle('Est. Power of the Largest Circle after Opening the Difference Matrix,k=[{0},{1}]'.format(s,s))
    f.savefig('pest-opening-diff-matrix-s{0}-{0}.png'.format(s))
####################################################################################################################################
## Different mas
####################################################################################################################################
## Trying opening on matricies directly
print("Applying opening to the difference matricies")
print("Trying opening")
pest_open = np.zeros(I.shape[2])
for ff in range(I.shape[2]):
    # erode then dilate
    open_diff = cv2.morphologyEx(I[:,:,ff].astype('uint8'),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # save opening preprocessing results
    cv2.imwrite(open_results_I + 'open-results-f{0}.png'.format(ff),open_diff)
    # search for circles in the image
    circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        pest_open[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

ax.clear()
ax.plot(pest_open)
ax.set(xlabel='Frame Index',ylabel='Est. Power (W)')
f.suptitle('Est. Power of the Largest Circle after Opening the Diff. Matrix')
f.savefig('pest-opening-diff-matrix.png')

print("Applying opening to the power density matricies")
print("Trying opening")
pest_open = np.zeros(I.shape[2])
for ff in range(I.shape[2]):
    # erode then dilate
    open_diff = cv2.morphologyEx(I[:,:,ff].astype('uint8'),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # save opening preprocessing results
    cv2.imwrite(open_results_I + 'open-results-I-f{0}.png'.format(ff),open_diff)
    # search for circles in the image
    circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        pest_open[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

ax.clear()
ax.plot(pest_open)
ax.set(xlabel='Frame Index',ylabel='Est. Power (W)')
f.suptitle('Est. Power of the Largest Circle after Opening the P.D. Matrix')
f.savefig('pest-opening-I-matrix.png')

####################################################################################################################################
## Looking at where the best circles consistently appears in the RAW difference matricies
print("Marking centres of interest on the RAW difference matricies")
best_ci_centres = np.zeros((rows,cols,3),np.uint8)
pdist_abs_centres = np.zeros(I.shape[2],dtype='float64')
best_centres = np.zeros((I.shape[2],2),dtype='float64')
for ff in range(I.shape[2]):
    # find difference matrix
    diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
    # find circles
    circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        # find the circle that is closest to 500.0
        pdist_all = np.array([np.abs(500.0-powerEstHoughCircle(c,I[:,:,ff],pixel_pitch)) for c in circles[0]])
        best_ci = pdist_all.argmin()
        # store distance between target circle and target
        pdist_abs_centres[ff] = pdist_all[best_ci]
        # store the centres
        best_centres[ff,:]=circles[0,best_ci,:2]
        # draw centre as a 1 radius circle on the mask
        cv2.circle(best_ci_centres, (circles[0,best_ci,0], circles[0,best_ci,1]), 1, (0,100,100), -1)

# plot which circle gave the best power estimate
ax.clear()
ax.imshow(best_ci_centres)
ax.set(xlabel='',ylabel='')
f.suptitle('Index of the Circles that Give the Best Estimate in the Difference Frame,Raw')
f.savefig('pest-best-circle-centres.png')

# find where the ABS power distance gets withinn 15 W
cc = best_centres[pdist_abs_centres<=15.0]
# create a blank mask
mask = np.zeros((rows,cols,3),np.uint8)
# draw the centre of each circle on the mask
for c in cc:
    cv2.circle(mask,(int(c[0]),int(c[1])),1,(0,100,100),-1)

ax.clear()
ax.imshow(mask)
f.suptitle('Circle Centres whose Power Estimates are within 15 Watts (Abs),Raw')
f.savefig('pbest-abs-best-circle-centres-15-watts.png')

## Looking at where the best circles consistently appears in the OPENED difference matricies
print("Marking centres of interest from the OPENED difference matricies")
best_ci_centres = np.zeros((rows,cols,3),np.uint8)
pdist_abs_centres = np.zeros(I.shape[2],dtype='float64')
best_centres = np.zeros((I.shape[2],2),dtype='float64')
for ff in range(I.shape[2]):
    # find difference matrix
    diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
    # apply erosion then dialation, opening
    open_diff = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # find circles
    circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        # find the circle that is closest to 500.0
        pdist_all = np.array([np.abs(500.0-powerEstHoughCircle(c,I[:,:,ff],pixel_pitch)) for c in circles[0]])
        best_ci = pdist_all.argmin()
        # store distance between target circle and target
        pdist_abs_centres[ff] = pdist_all[best_ci]
        # store the centres
        best_centres[ff,:]=circles[0,best_ci,:2]
        # draw centre as a 1 radius circle on the mask
        cv2.circle(best_ci_centres, (circles[0,best_ci,0], circles[0,best_ci,1]), 1, (0,100,100), -1)

ax.clear()
ax.imshow(best_ci_centres)
ax.set(xlabel='',ylabel='')
f.suptitle('Index of the Circles that Give the Best Estimate in the Frame, Opened')
f.savefig('pest-best-circle-centres-open.png')

# find where the ABS power distance gets withinn 15 W
cc = best_centres[pdist_abs_centres<=15.0]
# create a blank mask
mask = np.zeros((rows,cols,3),np.uint8)
# draw the centre of each circle on the mask
for c in cc:
    cv2.circle(mask,(int(c[0]),int(c[1])),1,(0,100,100),-1)

ax.clear()
ax.imshow(mask)
f.suptitle('Circle Centres whose Power Estimates are within 15 Watts (Abs)')
f.savefig('pbest-abs-best-circle-centres-15-watts-closed.png')

####################################################################################################################################
## Searching for the best circle after the results have been sorted by size
# the purpose of this is to prove that the largest circle in the given set is likely the best shot at choosing the closest power estimate
# if not, then it provides better search limits
print("Searching for the best circles after the circles have been sorted by size")
# using the raw difference matricies
print("Using RAW difference matricies")
fc,axc = plt.subplots()
best_ci = np.zeros(I.shape[2],np.uint16)
for ff in range(I.shape[2]):
   # find difference matrix
   diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
   # find circles
   circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
   if circles is not None:
      # sort by radius in descending order
      # therefor the largest is the first value in the matrix
      best_ci[ff] = circles[0,:,2].argsort()[::-1][0]

axc.plot(best_ci)
axc.set(xlabel='Frame Index',ylabel='Best Circle Indexbest_')
fc.suptitle('Best Circle Index in Raw Difference Matrix After Sorting')
fc.savefig('best-ci-sorted-diff-matrix.png')

# using the difference matricies after it has been opened
print("Using OPENED difference matricies")
best_ci = np.zeros(I.shape[2],np.uint16)
for ff in range(I.shape[2]):
   # find difference matrix
   diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
   # apply erosion then dialation, opening
   open_diff = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
   # find circles
   circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
   if circles is not None:
      # sort by radius in descending order
      # therefor the largest is the first value in the matrix
      best_ci[ff] = circles[0,:,2].argsort()[::-1][0]

axc.clear()
axc.plot(best_ci)
axc.set(xlabel='Frame Index',ylabel='Best Circle Index')
fc.suptitle('Best Circle Index in Opened Difference Matrix After Sorting')
fc.savefig('best-ci-sorted-opened-diff-matrix.png')

####################################################################################################################################
## Ratio between the radius of the circle closest to the target power and the laser radius
# way of investigating the laser radius at this height
print("Finding ratio between best circle and r0, RAW difference matrix")
fr,axr = plt.subplots()
bestc_rratio = np.zeros(I.shape[2],np.float64)
for ff in range(I.shape[2]):
    # find difference matrix
    diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
    # find circles
    circles = cv2.HoughCircles(diff_all,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        # find the circle that is closest to 500.0
        best_ci = np.array([np.abs(500.0-powerEstHoughCircle(c,I[:,:,ff],pixel_pitch)) for c in circles[0]]).argmin()
        # save ration between the best circle radius and the laser radius
        bestc_rratio[ff] = circles[0,best_ci,2]/r0_p

axr.plot(bestc_rratio)
axr.set(xlabel='Frame Index',ylabel='Ratio between Best Circle Radius and r0')
fr.suptitle('Ratio Between Radius of the Circle \n Closest to Target Power and Laser Radius, Raw')
fr.savefig('best-ci-rratio-r0-diff-matrix.png')

print("Finding ratio between best circle and r0, OPENED difference matrix")
bestc_rratio = np.zeros(I.shape[2],np.float64)
for ff in range(I.shape[2]):
    # find difference matrix
    diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
    open_diff = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    # find circles
    circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        # find the circle that is closest to 500.0
        best_ci = np.array([np.abs(500.0-powerEstHoughCircle(c,I[:,:,ff],pixel_pitch)) for c in circles[0]]).argmin()
        # save ration between the best circle radius and the laser radius
        bestc_rratio[ff] = circles[0,best_ci,2]/r0_p

axr.clear()
axr.plot(bestc_rratio)
axr.set(xlabel='Frame Index',ylabel='Ratio between Best Circle Radius and r0')
fr.suptitle('Ratio Between Radius of the Circle \n Closest to Target Power and Laser Radius, Opened')
fr.savefig('best-ci-rratio-r0-opened-diff-matrix.png')
        
####################################################################################################################################
# Use closing instead of opening
# closing dilates and then erodes
# DEV NOTE: Opening is more likely to be beneficial in this case as it removes noise and helps better identify the boundary
# Closing is more about closing small holes in the target foreground shape to clear it up
# It's still performed so all possibly helpful processes are investigated
print("Trying closing")
pest_close = np.zeros(I.shape[2])
for ff in range(I.shape[2]):
    diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
    # erode then dilate
    close_diff = cv2.morphologyEx(diff_all,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # save opening preprocessing results
    cv2.imwrite(close_results + 'close-results-f{0}.png'.format(ff),close_diff)
    # search for circles in the image
    circles = cv2.HoughCircles(close_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        pest_close[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

ax.clear()
ax.plot(pest_close)
ax.set(xlabel='Frame Index',ylabel='Est. Power (W)')
f.suptitle('Est. Power of the Largest Circle after Closing the Difference Matrix')
f.savefig('pest-closing-diff-matrix.png')

## Trying closing with different kernel sizes and saving the power plot
# still using the largest circle
fsize,axsize = plt.subplots()
for s in k_size:
    print("Trying closing, size ",s)
    pest_close = np.zeros(I.shape[2])
    for ff in range(I.shape[2]):
        diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
        # erode then dilate
        close_diff = cv2.morphologyEx(diff_all,cv2.MORPH_CLOSE,np.ones((s,s),np.uint8))
        # save opening preprocessing results
        cv2.imwrite(close_results + 'close-results-f{0}-s{1}-{1}.png'.format(ff,s),close_diff)
        # search for circles in the image
        circles = cv2.HoughCircles(close_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
        # if circles were found
        if circles is not None:
            # calculate power using the largest circle
            pest_close[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)
            
    axsize.clear()
    axsize.plot(pest_close)
    axsize.set(xlabel='Frame Index',ylabel='Est. Power (W)')
    fsize.suptitle('Est. Power of the Largest Circle \n after Closing the Difference Matrix,k=[{0},{1}]'.format(s,s))
    fsize.savefig('pest-closing-diff-matrix-s{0}-{0}.png'.format(s))

## Trying opening on the power density matricies directly
fclose,axclose = plt.subplots()
print("Applying closing to the power density matricies")
print("Trying closing")
pest_close = np.zeros(I.shape[2])
for ff in range(I.shape[2]):
    # erode then dilate
    close_diff = cv2.morphologyEx(I[:,:,ff].astype('uint8'),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # save opening preprocessing results
    tt = cv2.imwrite(close_results_I + 'close-results-I-f{0}.png'.format(ff),close_diff)
    # search for circles in the image
    circles = cv2.HoughCircles(close_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        pest_close[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

axclose.plot(pest_close)
axclose.set(xlabel='Frame Index',ylabel='Est. Power (W)')
fclose.suptitle('Est. Power of the Largest Circle after Opening the P.D. Matrix')
fclose.savefig('pest-closing-I-matrix.png')

####################################################################################################################################
## Looking at where the best circles consistently appears in the CLOSED difference matricies
print("Marking centres of interest on the CLOSED difference matricies")
best_ci_centres = np.zeros((rows,cols,3),np.uint8)
pdist_abs_centres = np.zeros(I.shape[2],dtype='float64')
best_centres = np.zeros((I.shape[2],2),dtype='float64')
for ff in range(I.shape[2]):
    # find difference matrix
    diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
    # apply erosion then dialation, opening
    close_diff = cv2.morphologyEx(diff_all,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # find circles
    circles = cv2.HoughCircles(close_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        # find the circle that is closest to 500.0
        pdist_all = np.array([np.abs(500.0-powerEstHoughCircle(c,I[:,:,ff],pixel_pitch)) for c in circles[0]])
        best_ci = pdist_all.argmin()
        # store distance between target circle and target
        pdist_abs_centres[ff] = pdist_all[best_ci]
        # store the centres
        best_centres[ff,:]=circles[0,best_ci,:2]
        # draw centre as a 1 radius circle on the mask
        cv2.circle(best_ci_centres, (circles[0,best_ci,0], circles[0,best_ci,1]), 1, (0,100,100), -1)

ax.clear()
ax.imshow(best_ci_centres)
ax.set(xlabel='',ylabel='')
f.suptitle('Index of the Circles that Give the Best Estimate in the Frame, Closed')
f.savefig('pest-best-circle-centres-closed.png')

# find where the ABS power distance gets withinn 15 W
cc = best_centres[pdist_abs_centres<=15.0]
# create a blank mask
mask = np.zeros((rows,cols,3),np.uint8)
# draw the centre of each circle on the mask
for c in cc:
    cv2.circle(mask,(int(c[0]),int(c[1])),1,(0,100,100),-1)

ax.clear()
ax.imshow(mask)
f.suptitle('Circle Centres whose Power Estimates are within 15 Watts (Abs),Closed')
f.savefig('pbest-abs-best-circle-centres-15-watts-closed.png')

####################################################################################################################################
### Trying kernels of different shapes
### Previous runs use a rectangular kernel but different ones are available
## Different kernel shapes
# size of each one, square size
k_size = 5
# square
square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(k_size,k_size))
# ellipse
ellip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k_size,k_size))
# cross
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(k_size,k_size))
# circle
# getStructuringElement does not have a circle option so it has to be created manually
mask = np.zeros((k_size,k_size),np.uint8)
# draw filled circles on blank mask
# the floor commands are so that it correctly builds the ksize circle
# e.g. floor(5/2) = 2
circle_kernel = cv2.circle(mask,(int(np.floor(k_size/2)),int(np.floor(k_size/2))),int(np.floor(k_size/2)),(1),-1)

# create dictionary so it can be iterated through
kernel_dict = {"ellipse" : ellip_kernel,
               "cross" : cross_kernel,
               "circle" : circle_kernel,
               "rectangle" : square_kernel}

# new axes for results
f,ax = plt.subplots()
# list for Avg. distance to 500W 
avg_dist = [0.0]*len(kernel_dict)
# iterate through each kernel
print("Trying kernels of different shapes, diff matrix")
for ki,(kname,kernel) in enumerate(kernel_dict.items()):
   print("Trying ",kname,"...")
   # clear power estimate array
   pest_open = np.zeros(I.shape[2])
   # iterate through each frame
   for ff in range(I.shape[2]):
       diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
       # erode then dilate
       open_diff = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,kernel)
       # save opening preprocessing results
       cv2.imwrite(shape_results + 'open-results-k-{0}-f{1}.png'.format(kname,ff),open_diff)
       # search for circles in the image
       circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
       if circles is not None:
           pest_open[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

   ax.clear()
   ax.plot(pest_open)
   ax.set(xlabel='Frame Index',ylabel='Est. Power (W)')
   f.suptitle('Est. Power of the Largest Circle \n after Opening with {0} Kernel, Diff. Matrix'.format(kname))
   f.savefig('pest-opening-k-{0}-diff-matrix.png'.format(kname))

   # Avg. distance between power profile and target power
   avg_dist[ki]=(np.abs(500.0-pest_open).mean())

# plot Avg. distance against the different kernel shapes
ax.clear()
ax.plot(list(kernel_dict.keys()),avg_dist,'r*')
ax.set(xlabel='Kernal Shape',ylabel='Avg. Distance to 500.0 (W)')
f.suptitle('Avg. Distance Between Power Profile \n and 500.0 W using different Kernel Shapes')
f.savefig('avg-pdist-kernel-shapes-diff-matrix.png')

# list for Avg. distance to 500W 
avg_dist = [0.0]*len(kernel_dict)
# iterate through each kernel
print("Trying kernels of different shapes, power density matricies")
for ki,(kname,kernel) in enumerate(kernel_dict.items()):
   print("Trying ",kname,"...")
   # clear power estimate array
   pest_open = np.zeros(I.shape[2])
   # iterate through each frame
   for ff in range(I.shape[2]):
       # erode then dilate
       open_diff = cv2.morphologyEx(I[:,:,ff].astype('uint8'),cv2.MORPH_OPEN,kernel)
       # save opening preprocessing results
       cv2.imwrite(shape_results + 'open-results-I-k-{0}-f{1}.png'.format(kname,ff),open_diff)
       # search for circles in the image
       circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
       if circles is not None:
           pest_open[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

   ax.clear()
   ax.plot(pest_open)
   ax.set(xlabel='Frame Index',ylabel='Est. Power (W)')
   f.suptitle('Est. Power of the Largest Circle \n after Opening with {0} Kernel, P.D. Matrix'.format(kname))
   f.savefig('pest-opening-I-k-{0}-diff-matrix.png'.format(kname))

   # Avg. distance between power profile and target power
   avg_dist[ki]=(np.abs(500.0-pest_open).mean())

# plot Avg. distance against the different kernel shapes
ax.clear()
ax.plot(list(kernel_dict.keys()),avg_dist,'r*')
ax.set(xlabel='Kernal Shape',ylabel='Avg. Distance to 500.0 (W)')
f.suptitle('Avg. Distance Between Power Profile \n and 500.0 W using different Kernel Shapes')
f.savefig('avg-pdist-I-kernel-shapes-diff-matrix.png')
####################################################################################################################################
### Trying different number of iterations
print("Trying different iterations of erosion and dialating")
erode_mesh,dilate_mesh = np.mgrid[1:10:1,1:10:1]
it_vals = np.vstack([erode_mesh.ravel(),dilate_mesh.ravel()])
# 2x2 kernel is used so that it has a different impact each iteration
# if it was 3x3 then the centre is in the actual. the same result is returned each convolution
k = np.ones((2,2),np.uint8)
pdist_all = np.zeros(erode_mesh.shape)
# iterate through pairs of iteration combinations
for ii in range(it_vals.shape[1]):
   # apply iteration combinations to each frame
   pest_all = np.zeros(I.shape[2])
   for ff in range(I.shape[2]):
      # calculate difference matrix
      diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
      # perform eroding and closing in different iterations
      res = customOpenAction(diff_all,k,erode_it=it_vals[0,ii],dilate_it=it_vals[1,ii])
      # search for circles in the image
      circles = cv2.HoughCircles(res,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
      # if circles were found
      if circles is not None:
        # Calculate power using the largest circle
        pest_all[ff]=np.abs(500.0-powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch))
   # store the Avg. of the distance between the power profile and the target of 500.0 Watts
   pdist_all[np.unravel_index(ii,pdist_all.shape)]=pest_all.mean()

# create axes
fct,axct = plt.subplots()
# draw results as contour
dist_ct = axct.contourf(pdist_all)
# update title and axes
axct.set(xlabel='Number of Erode Iterations',ylabel='Number of Dilate Iterations')
fct.suptitle('Distance to Target Power Using Largest Circle and Different Iterations')
# add color bar showing range of results
div = make_axes_locatable(axct)
cax = div.append_axes("right",size="7%",pad="2%")
colorbar(dist_ct,cax=cax)
fct.savefig('avg-pdist-erode-dilate-iter.png')

####################################################################################################################################
### Opening and then Closing
# raw values
print("Trying opening then closing, raw values")
pest_open = np.zeros(I.shape[2])
for ff in range(I.shape[2]):
    diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
    # erode then dilate
    open_diff = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    close_diff = cv2.morphologyEx(open_diff,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # save opening preprocessing results
    cv2.imwrite(close_open_results + 'open-close-results-f{0}.png'.format(ff),close_diff)
    # search for circles in the image
    circles = cv2.HoughCircles(close_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        pest_open[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

ax.plot(pest_open)
ax.set(xlabel='Frame Index',ylabel='Est. Power (W)')
f.suptitle('Est. Power of the Largest Circle after Opening the Difference Matrix')
f.savefig('pest-open-close-diff-matrix.png')

# power density values
print("Trying opening then closing, power density values")
pest_open = np.zeros(I.shape[2])
for ff in range(I.shape[2]):
    # erode then dilate
    open_diff = cv2.morphologyEx(I[:,:,ff].astype('uint8'),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    close_diff = cv2.morphologyEx(open_diff,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    # save opening preprocessing results
    cv2.imwrite(close_open_results + 'open-close-results-I-f{0}.png'.format(ff),close_diff)
    # search for circles in the image
    circles = cv2.HoughCircles(close_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
    if circles is not None:
        pest_open[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)

ax.clear()
ax.plot(pest_open)
ax.set(xlabel='Frame Index',ylabel='Est. Power (W)')
f.suptitle('Est. Power of the Largest Circle after Opening the Difference Matrix')
f.savefig('pest-open-close-power-density.png')
####################################################################################################################################
### Filtering circle sizes
print("Filtering circles smaller than r0 and greater than a set limit")
upper_lim = np.arange(1.0,3.4,0.1)
flim,axlim = plt.subplots()
print("Trying different max upper radius, opened diff")
for lim in upper_lim:
   # create empty matrix for power estimate
   pest = np.zeros(I.shape[2])
   for ff in range(I.shape[2]):
      # calculate difference matrix
      diff_all = (diffRow(I[:,:,ff]) + diffCol(I[:,:,ff])).astype('uint8')
      open_diff = cv2.morphologyEx(diff_all,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
      # search for circles in the image
      circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
      # if circles were found
      if circles is not None:
         # mask circle radii that are below r0 and above r0*lim
         circles = maskCirclesRadius(circles,rmin=r0_p,rmax=r0_p*lim)
         # if there are circles left
         if circles.shape[1]>0:
            # calculate power using the largest circle
            pest[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)
   
   axlim.clear()
   axlim.plot(pest)
   axlim.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
   flim.suptitle('Power Estimate using Largest Circle After Removing Circles r>{:.2f}xr0'.format(lim))
   flim.savefig('pest-radii-filter-l{0}-{1}-diff-matrix.png'.format(*'{:.2f}'.format(lim).split('.')))

print("Trying different max upper radius, opened I")
for lim in upper_lim:
   # create empty matrix for power estimate
   pest = np.zeros(I.shape[2])
   for ff in range(I.shape[2]):
      # calculate difference matrix
      open_diff = cv2.morphologyEx(I[:,:,ff].astype('uint8'),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
      # search for circles in the image
      circles = cv2.HoughCircles(open_diff,cv2.HOUGH_GRADIENT,1,rows/8,param1=100,param2=30,minRadius=0,maxRadius=rmax_gauss)
      # if circles were found
      if circles is not None:
         # mask circle radii that are below r0 and above r0*lim
         circles = maskCirclesRadius(circles,rmin=r0_p,rmax=r0_p*lim)
         # if there are circles left
         if circles.shape[1]>0:
            # calculate power using the largest circle
            pest[ff]=powerEstHoughCircle(circles[0,circles[0,:,2].argmax(),:],I[:,:,ff],pixel_pitch)
   
   axlim.clear()
   axlim.plot(pest)
   axlim.set(xlabel='Frame Index',ylabel='Power Estimate (W)')
   flim.suptitle('Power Estimate using Largest Circle After Removing Circles r>{:.2f}xr0'.format(lim))
   flim.savefig('pest-radii-filter-l{0}-{1}-I-matrix.png'.format(*'{:.2f}'.format(lim).split('.')))
